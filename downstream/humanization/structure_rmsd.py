import os
import glob
import csv
import numpy as np
import abnumber
from Bio import PDB
from Bio.SeqUtils import seq1

# ================== 核心 RMSD 计算逻辑 (保持不变) ==================

def extract_chain_info(cif_path, chain_id, scheme="imgt"):
    """
    1. 读取CIF
    2. 使用 abnumber 解析 V 区
    3. 通过 fr1_seq, cdr1_seq 等属性构建 Mask
    4. 裁切坐标并返回
    """
    parser = PDB.MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("struct", cif_path)
        chain = structure[0][chain_id]
    except KeyError:
        raise KeyError(f"Chain {chain_id} not found in {cif_path}")
    except Exception as e:
        raise ValueError(f"Biopython parse error for {cif_path}: {e}")

    # 提取完整序列和坐标
    residues = [r for r in chain if PDB.is_aa(r)]
    full_seq = "".join([seq1(r.get_resname()) for r in residues])
    full_coords = np.array([r['CA'].get_coord() for r in residues])

    # 解析 V 区
    try:
        an_chain = abnumber.Chain(full_seq, scheme=scheme)
    except Exception:
        # 尝试作为轻链或重链再次解析，或者报错
        raise ValueError(f"Abnumber failed to parse chain {chain_id} in {os.path.basename(cif_path)}")

    # 基于属性构建 Mask 和 V区序列 (顺序: FR1-CDR1-FR2-CDR2-FR3-CDR3-FR4)
    regions = [
        (an_chain.fr1_seq,  'FR'),
        (an_chain.cdr1_seq, 'CDR'),
        (an_chain.fr2_seq,  'FR'),
        (an_chain.cdr2_seq, 'CDR'),
        (an_chain.fr3_seq,  'FR'),
        (an_chain.cdr3_seq, 'CDR3'), # 目标区域
        (an_chain.fr4_seq,  'FR')
    ]

    v_seq_constructed = ""
    fr_mask_list = []
    cdr3_mask_list = []

    for seq_segment, region_type in regions:
        if seq_segment is None: seq_segment = ""
        
        length = len(seq_segment)
        v_seq_constructed += seq_segment
        
        # 构建 Mask
        if region_type == 'FR':
            fr_mask_list.extend([True] * length)
            cdr3_mask_list.extend([False] * length)
        elif region_type == 'CDR3':
            fr_mask_list.extend([False] * length)
            cdr3_mask_list.extend([True] * length)
        else: # CDR1, CDR2 (不用于对齐，也不用于计算RMSD)
            fr_mask_list.extend([False] * length)
            cdr3_mask_list.extend([False] * length)

    fr_mask = np.array(fr_mask_list, dtype=bool)
    cdr3_mask = np.array(cdr3_mask_list, dtype=bool)

    # 坐标裁切
    start_idx = full_seq.find(v_seq_constructed)
    if start_idx == -1:
        raise ValueError(f"Parsed V-sequence not found in original sequence (Chain: {chain_id})")
    
    end_idx = start_idx + len(v_seq_constructed)
    fv_coords = full_coords[start_idx:end_idx]

    return fv_coords, fr_mask, cdr3_mask

def calc_antibody_rmsd(gt_path, pred_path, gt_heavy_id, gt_light_id, pred_heavy_id, pred_light_id):
    # 1. 提取数据
    gt_h, gt_h_fr, gt_h_cdr3 = extract_chain_info(gt_path, gt_heavy_id)
    gt_l, gt_l_fr, _         = extract_chain_info(gt_path, gt_light_id)

    pred_h, _, _ = extract_chain_info(pred_path, pred_heavy_id)
    pred_l, _, _ = extract_chain_info(pred_path, pred_light_id)

    # 2. 长度校验
    if len(gt_h) != len(pred_h): 
        raise ValueError(f"Heavy chain Fv length mismatch: GT={len(gt_h)}, Pred={len(pred_h)}")
    if len(gt_l) != len(pred_l): 
        raise ValueError(f"Light chain Fv length mismatch: GT={len(gt_l)}, Pred={len(pred_l)}")
    if not np.any(gt_h_cdr3): 
        raise ValueError("No CDR3 found in GT Heavy chain")

    # 3. 对齐 (Align on Combined FR)
    # GT FR Points
    gt_fr_pts = np.concatenate([gt_h[gt_h_fr], gt_l[gt_l_fr]])
    # Pred FR Points
    pred_fr_pts = np.concatenate([pred_h[gt_h_fr], pred_l[gt_l_fr]])

    # Kabsch Algorithm
    c_gt = gt_fr_pts.mean(axis=0)
    c_pred = pred_fr_pts.mean(axis=0)
    Q = gt_fr_pts - c_gt
    P = pred_fr_pts - c_pred
    H = np.dot(P.T, Q)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # 4. 计算 H-CDR3 RMSD
    target_gt = gt_h[gt_h_cdr3]
    target_pred_raw = pred_h[gt_h_cdr3]
    
    target_pred_aligned = np.dot(target_pred_raw - c_pred, R.T) + c_gt
    
    diff = target_pred_aligned - target_gt
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    return rmsd

# ================== 数据读取与批处理逻辑 ==================

def parse_mapping_csv(csv_path):
    """
    解析 CSV 文件，格式: 8tq8,H-L
    返回字典: {'8tq8': {'H': 'H', 'L': 'L'}, ...}
    """
    mapping = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            pdb_id = row[0].strip()
            chains = row[1].strip() # 格式如 "H-L" 或 "B-A"
            
            try:
                h_chain, l_chain = chains.split('-')
                mapping[pdb_id] = {'H': h_chain, 'L': l_chain}
            except ValueError:
                print(f"Warning: Skipping invalid CSV format for {pdb_id}: {chains}")
    return mapping

def find_prediction_cif(pred_root_dir, pdb_id):
    """
    1. 在 pred_root_dir 中找到以 fold_{pdb_id} 开头的子目录
    2. 在子目录中找到 .cif 文件
    3. 返回第一个找到的 cif 文件路径 (通常是 model_0 或 rank_0)
    """
    # 1. 查找子目录 pattern: fold_8tq8_*
    # 使用 glob 查找目录
    dir_pattern = os.path.join(pred_root_dir, f"fold_{pdb_id}*")
    subdirs = glob.glob(dir_pattern)
    
    if not subdirs:
        return None, f"No folder found for fold_{pdb_id}"
    
    # 取第一个匹配的文件夹
    target_dir = subdirs[0]
    
    # 2. 查找 CIF 文件
    cif_pattern = os.path.join(target_dir, "*.cif")
    cif_files = glob.glob(cif_pattern)
    
    if not cif_files:
        return None, f"No .cif files found inside {target_dir}"
    
    # 3. 排序并返回第一个 (模拟 '直接用预测出来的第一个结构')
    # 通常 AlphaFold 输出是按 model_0, model_1... 排序的
    cif_files.sort()
    return cif_files[0], None

def main():
    # ============ 配置路径 ============
    # Ground Truth 所在的文件夹
    ground_truth_dir = "/mnt/nas-new/home/yangnianzu/jm/bjzgc_zyh/AirGen-Dev/data/humanization/data/humanisation/test-pdb" 
    
    # 包含 ID 和 链映射的 CSV 文件路径
    mapping_csv_path = "/mnt/nas-new/home/yangnianzu/jm/bjzgc_zyh/AirGen-Dev/data/humanization/data/humanisation/test_chains.csv" 
    
    # 预测结果的大文件夹 (包含 fold_xxxx 子文件夹)
    predict_root_dir = "/mnt/nas-new/home/yangnianzu/jm/bjzgc_zyh/AirGen-Dev/data/prediction" 
    
    # 预测文件的默认链 ID (通常 AF 输出 Heavy=A, Light=B，根据你的输入顺序决定)
    pred_heavy_id_default = "A"
    pred_light_id_default = "B"
    # =================================
    
    print(f"{'PDB_ID':<10} | {'GT_Chains':<10} | {'Result':<20} | {'RMSD (Å)':<10}")
    print("-" * 60)

    # 1. 读取映射
    gt_mapping = parse_mapping_csv(mapping_csv_path)
    
    results = []

    # 2. 遍历每个 PDB ID 进行计算
    for pdb_id, chains in gt_mapping.items():
        gt_h_id = chains['H']
        gt_l_id = chains['L']
        
        # 2.1 构建 GT 文件路径
        gt_file = os.path.join(ground_truth_dir, f"{pdb_id}.cif")
        if not os.path.exists(gt_file):
            print(f"{pdb_id:<10} | {gt_h_id}-{gt_l_id:<10} | {'GT File Missing':<20} | N/A")
            continue

        # 2.2 查找 Pred 文件路径
        pred_file, err = find_prediction_cif(predict_root_dir, pdb_id)
        if pred_file is None:
            print(f"{pdb_id:<10} | {gt_h_id}-{gt_l_id:<10} | {err:<20} | N/A")
            continue

        # 2.3 计算 RMSD
        try:
            rmsd_val = calc_antibody_rmsd(
                gt_path=gt_file,
                pred_path=pred_file,
                gt_heavy_id=gt_h_id,
                gt_light_id=gt_l_id,
                pred_heavy_id=pred_heavy_id_default, # 假设预测文件 H=A
                pred_light_id=pred_light_id_default  # 假设预测文件 L=B
            )
            
            print(f"{pdb_id:<10} | {gt_h_id}-{gt_l_id:<10} | {'Success':<20} | {rmsd_val:.4f}")
            results.append((pdb_id, rmsd_val))
            
        except Exception as e:
            # 捕获错误 (如长度不一致、解析失败等)，不中断循环
            error_msg = str(e).split('\n')[0] # 只取第一行错误信息
            # 缩短错误信息以便打印
            if len(error_msg) > 20: error_msg = "Calc Error"
            print(f"{pdb_id:<10} | {gt_h_id}-{gt_l_id:<10} | {error_msg:<20} | N/A")
            # 可选: 打印详细堆栈用于调试
            # import traceback; traceback.print_exc()

    # 3. 计算平均 RMSD
    if results:
        valid_rmsds = [r[1] for r in results]
        avg_rmsd = np.mean(valid_rmsds)
        print("-" * 60)
        print(f"Average H-CDR3 RMSD: {avg_rmsd:.4f} Å (Count: {len(valid_rmsds)})")

if __name__ == "__main__":
    main()