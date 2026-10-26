"""Export panel a as native editable draw.io cells using the shared figure layout.

Run after activating conda pllm:
    python scripts/export_research_architecture_drawio.py

Also regenerates the matching panel-a PNG, SVG and PDF. No image is embedded
in the draw.io file: rectangles, text and connectors remain separate objects.
"""
from html import escape
import xml.etree.ElementTree as ET

from matplotlib.colors import to_hex, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.text import Text

from draw_research_refined_paper import OUT, draw


MATH_LABELS = {
    r'H / $\beta$': 'H / β',
    r'L / $\alpha$': 'L / α',
    r'$E_{\mathrm{token}} + E_{\mathrm{role}} + E_{\mathrm{scope}}$':
        'E<sub>token</sub> + E<sub>role</sub> + E<sub>scope</sub>',
    r'$p_\theta(x_0\,|\,x_t,c)$': 'p<sub>θ</sub>(x<sub>0</sub> | x<sub>t</sub>, c)',
}


def color(value):
    rgba = to_rgba(value)
    return 'none' if rgba[3] == 0 else to_hex(rgba)


def export():
    fig = draw(panel='a')
    fig.canvas.draw()
    ax = fig.axes[0]
    renderer = fig.canvas.get_renderer()
    mxfile = ET.Element('mxfile', host='app.diagrams.net', type='device')
    diagram = ET.SubElement(mxfile, 'diagram', id='immune-architecture-a', name='Architecture')
    model = ET.SubElement(diagram, 'mxGraphModel', grid='0', guides='1', tooltips='1',
                          connect='1', arrows='1', fold='1', page='1', pageScale='1',
                          pageWidth='1440', pageHeight='440', math='0', shadow='0',
                          background='#ffffff')
    root = ET.SubElement(model, 'root')
    ET.SubElement(root, 'mxCell', id='0')
    ET.SubElement(root, 'mxCell', id='1', parent='0')
    count = 0

    def vertex(x, y, w, h, style, value=''):
        nonlocal count
        count += 1
        cell = ET.SubElement(root, 'mxCell', id=f'v{count}', value=value,
                             style=style, vertex='1', parent='1')
        ET.SubElement(cell, 'mxGeometry', x=f'{x:.3f}', y=f'{y:.3f}',
                      width=f'{w:.3f}', height=f'{h:.3f}', attrib={'as': 'geometry'})

    def edge(points, stroke, width, arrow=False, dashed=False):
        nonlocal count
        count += 1
        style = (f'edgeStyle=none;rounded=0;html=1;strokeColor={stroke};'
                 f'strokeWidth={width:.3f};endArrow={"block" if arrow else "none"};'
                 f'endFill=1;endSize=7;dashed={int(dashed)};')
        cell = ET.SubElement(root, 'mxCell', id=f'e{count}', style=style, edge='1', parent='1')
        geo = ET.SubElement(cell, 'mxGeometry', relative='1', attrib={'as': 'geometry'})
        for point, kind in [(points[0], 'sourcePoint'), (points[-1], 'targetPoint')]:
            ET.SubElement(geo, 'mxPoint', x=str(point[0]), y=str(point[1]), attrib={'as': kind})
        if len(points) > 2:
            waypoints = ET.SubElement(geo, 'Array', attrib={'as': 'points'})
            for x, y in points[1:-1]:
                ET.SubElement(waypoints, 'mxPoint', x=str(x), y=str(y))

    artists = sorted([*ax.patches, *ax.lines, *ax.texts], key=lambda a: a.get_zorder())
    for artist in artists:
        if isinstance(artist, Text):
            if not 0 <= artist.get_position()[1] <= 440:
                continue
            bounds = artist.get_window_extent(renderer)
            x0, y1 = ax.transData.inverted().transform((bounds.x0, bounds.y0))
            x1, y0 = ax.transData.inverted().transform((bounds.x1, bounds.y1))
            raw = artist.get_text()
            label = MATH_LABELS.get(raw, escape(raw).replace('\n', '<br>'))
            if '$' in label:
                raise ValueError(f'Unconverted math label: {raw}')
            fs = artist.get_fontsize() * 100 / 72
            style = ('text;html=1;strokeColor=none;fillColor=none;whiteSpace=nowrap;'
                     'overflow=visible;align=center;verticalAlign=middle;spacing=0;'
                     f'fontFamily=DejaVu Sans;fontSize={fs:.3f};'
                     f'fontColor={color(artist.get_color())};'
                     f'fontStyle={1 if artist.get_weight() == "bold" else 0};')
            vertex(x0-2, y0-1, x1-x0+4, y1-y0+2, style, label)
        elif isinstance(artist, FancyArrowPatch):
            points = artist._posA_posB
            if max(p[1] for p in points) <= 440:
                edge(points, color(artist.get_edgecolor()), artist.get_linewidth()*100/72, True)
        elif isinstance(artist, Line2D):
            points = list(zip(artist.get_xdata(), artist.get_ydata()))
            if max(p[1] for p in points) <= 440:
                edge(points, color(artist.get_color()), artist.get_linewidth()*100/72,
                     dashed=artist.is_dashed())
        elif isinstance(artist, (Rectangle, Circle)):
            if isinstance(artist, Circle):
                r = artist.radius
                x, y = artist.center[0]-r, artist.center[1]-r
                w = h = r*2
                shape = 'ellipse'
            else:
                x, y, w, h = artist.get_x(), artist.get_y(), artist.get_width(), artist.get_height()
                shape = 'rectangle'
            if y >= 440 or y+h < 0:
                continue
            style = (f'shape={shape};rounded=0;html=1;fillColor={color(artist.get_facecolor())};'
                     f'strokeColor={color(artist.get_edgecolor())};'
                     f'strokeWidth={artist.get_linewidth()*100/72:.3f};')
            vertex(x, y, w, h, style)

    ET.indent(mxfile, space='  ')
    path = OUT / 'research_refined_architecture_a.drawio'
    ET.ElementTree(mxfile).write(path, encoding='utf-8', xml_declaration=True)
    parsed = ET.parse(path)
    cells = parsed.findall('.//mxCell')
    assert len({cell.get('id') for cell in cells}) == len(cells)
    assert len(parsed.findall('.//mxCell[@edge="1"]')) > 5
    assert len(parsed.findall('.//mxCell[@vertex="1"]')) > 50
    assert 'image=' not in path.read_text()
    print(f'{path}: {count} editable objects; XML validation passed')


if __name__ == '__main__':
    export()
