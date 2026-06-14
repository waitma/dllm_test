import os
import pandas as pd



def main():
    df_dir = '/mnt/nas-new/home/yangnianzu/jm/bjzgc_zyh/AirGen-Dev/generate_results/airgen_gumbel_argmax_64_results_n100.csv'
    pairing_df_dir = f"{df_dir}_gen_l_sequence_pairing_scores.csv"
    pairing_df = pd.read_csv(pairing_df_dir)

    k_pairing_df = pairing_df[pairing_df['gen_light_type'] == 'kappa']
    l_pairing_df = pairing_df[pairing_df['gen_light_type'] == 'lambda']

    score_k = k_pairing_df['pairing_scores'].mean()
    score_l = l_pairing_df['pairing_scores'].mean()

    print(f"Score K: {score_k}")
    print(f"Score L: {score_l}")


if __name__ == "__main__":
    main()