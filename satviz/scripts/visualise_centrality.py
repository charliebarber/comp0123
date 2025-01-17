import networkx as nx
import pickle

NAME="Kuiper-nx"

# General files needed to generate visualizations; Do not change for different simulations
topFile = "../static_html/top.html"
bottomFile = "../static_html/bottom.html"

# Output directory for creating visualization html files
OUT_DIR = "../viz_output/"
# JSON_NAME  = NAME+"_5shell.json"
# OUT_JSON_FILE = OUT_DIR + JSON_NAME
OUT_HTML_FILE = OUT_DIR + NAME + ".html"

def visualise_graph(G):
    viz_string = ""
    
    # Add satellites as black spheres
    for node_id, node_data in G.nodes(data=True):
        if node_data.get('type') == 'satellite':
            viz_string += (
                "var satellite = viewer.entities.add({\n"
                "    name : '',\n"
                f"    position: Cesium.Cartesian3.fromDegrees({node_data['long_deg']}, "
                f"{node_data['lat_deg']}, {node_data['altitude']}),\n"
                "    ellipsoid : {\n"
                "        radii : new Cesium.Cartesian3(30000.0, 30000.0, 30000.0),\n"
                "        material : Cesium.Color.BLACK.withAlpha(1),\n"
                "    }\n"
                "});\n"
            )
        elif node_data.get('type') == 'ground_station':
            viz_string += (
                "var gs = viewer.entities.add({\n"
                "    name : '',\n"
                f"    position: Cesium.Cartesian3.fromDegrees({node_data['long_str']}, "
                f"{node_data['lat_str']}, 0),\n"
                "    ellipsoid : {\n"
                "        radii : new Cesium.Cartesian3(30000.0, 30000.0, 30000.0),\n"
                "        material : Cesium.Color.BLUE.withAlpha(1),\n"
                "    }\n"
                "});\n"
            )
    
    # Add edges as blue lines
    for edge in G.edges():
        node1_data = G.nodes[edge[0]]
        node2_data = G.nodes[edge[1]]
        
        if (node1_data.get('type') == 'satellite' and 
            node2_data.get('type') == 'satellite'):
            
            viz_string += (
                "viewer.entities.add({\n"
                "    name : '',\n"
                "    polyline: {\n"
                "        positions: Cesium.Cartesian3.fromDegreesArrayHeights([\n"
                f"            {node1_data['long_deg']},\n"
                f"            {node1_data['lat_deg']},\n"
                f"            {node1_data['altitude']},\n"
                f"            {node2_data['long_deg']},\n"
                f"            {node2_data['lat_deg']},\n"
                f"            {node2_data['altitude']}\n"
                "        ]),\n"
                "        width: 0.5,\n"
                "        arcType: Cesium.ArcType.NONE,\n"
                "        material: new Cesium.PolylineOutlineMaterialProperty({\n"
                "            color: Cesium.Color.DODGERBLUE.withAlpha(0.4),\n"
                "            outlineWidth: 0,\n"
                "            outlineColor: Cesium.Color.BLACK\n"
                "        })\n"
                "    }\n"
                "});\n"
            )
    
    return viz_string


def write_viz_files(vis_string):
    """
    Writes JSON and TML files to the output folder
    :return: None
    """
    writer_html = open(OUT_HTML_FILE, 'w')
    with open(topFile, 'r') as fi:
        writer_html.write(fi.read())
    writer_html.write(viz_string)
    with open(bottomFile, 'r') as fb:
        writer_html.write(fb.read())
    writer_html.close()

# with open("../../analysis/data/starlink_550/2000ms_for_200s/pickles/0.pickle", "rb") as f:
# with open("../../analysis/data/kuiper_630/2000ms_for_200s/pickles/2000000000.pickle", "rb") as f:
with open("../../analysis/data/kuiper_630/100ms_for_1s/pickles/0.pickle", "rb") as f:
    G = pickle.load(f)
    viz_string = visualise_graph(G)
    write_viz_files(viz_string)