import json
import matplotlib.pyplot as plt
import re
import os
import sys

home_path = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
print(home_path)
if home_path not in sys.path:
    sys.path.append(home_path)

import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages

# setting font globally
font_path = home_path + '/visualization/fonts/Times New Roman.ttf'
font = fm.FontProperties(fname=font_path, size=12)

import json
import matplotlib.pyplot as plt
import re
import os

def plot_pose_estimation_results(input_path, output_file_path, dataset_name, model_name):
    # Initialize lists for storing data by method
    methods_data = {}
    
    # Colors and markers for different methods (expand if more methods are added)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']  
    markers = ['o', 'x', '^', 's', 'D', '*']  

    # Loop through each folder in the input_path
    for folder_name in os.listdir(input_path):
        folder_path = os.path.join(input_path, folder_name)
        # Check if it’s a directory
        if os.path.isdir(folder_path):
            # Parse method name, rotation, and translation thresholds from folder name
            match = re.match(r'([A-Za-z]+)_testset_r_(\d+\.\d+)_t_(\d+\.\d+)', folder_name)
            if match:
                method_name = match.group(1)
                rotation_threshold = float(match.group(2))
                translation_threshold = float(match.group(3))
                
                # Load the JSON file inside the folder
                for file in os.listdir(folder_path):
                    if file.endswith('.json'):
                        json_path = os.path.join(folder_path, file)
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                            valid_percentage = data["valid_percentage"]
                        
                        # Initialize method data list if not already present
                        if method_name not in methods_data:
                            methods_data[method_name] = {
                                'rotation_data': [],
                                'translation_data': []
                            }
                        
                        # Append data to corresponding method
                        methods_data[method_name]['rotation_data'].append((rotation_threshold, valid_percentage))
                        methods_data[method_name]['translation_data'].append((translation_threshold, valid_percentage))
    
    # Sort each method's data by threshold values
    for method_name, data in methods_data.items():
        data['rotation_data'].sort()
        data['translation_data'].sort()

    # Create a separate plot for Rotation Error Threshold vs Valid Percentage
    plt.figure(figsize=(7, 6))
    for i, (method_name, data) in enumerate(methods_data.items()):
        rot_x, rot_y = zip(*data['rotation_data'])
        plt.plot(rot_x, rot_y, marker=markers[i % len(markers)], color=colors[i % len(colors)], label=method_name)
    # plt.title('Valid Percentage vs Rotation Error Threshold', fontproperties=font)
    plt.xlabel('Rotation Error Threshold (degree)', fontproperties=font)
    plt.ylabel('Success Rate (%)', fontproperties=font)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Save the plot to a PNG file and pdf file
    plt.savefig(output_file_path + '/Rotation_Error_Success_Rate_{}_{}.svg'.format(dataset_name, model_name), format="svg")
    plt.savefig(output_file_path + '/Rotation_Error_Success_Rate_{}_{}.png'.format(dataset_name, model_name), bbox_inches='tight', dpi=500, pad_inches=0.1)
    plt.savefig(output_file_path + '/Rotation_Error_Success_Rate_{}_{}.eps'.format(dataset_name, model_name), format="eps")
    pp = PdfPages(output_file_path + '/Rotation_Error_Success_Rate_{}_{}.pdf'.format(dataset_name, model_name))
    pp.savefig()
    pp.close()
    plt.close()
    
    # Create a separate plot for Translation Error Threshold vs Valid Percentage
    plt.figure(figsize=(7, 6))
    for i, (method_name, data) in enumerate(methods_data.items()):
        trans_x, trans_y = zip(*data['translation_data'])
        plt.plot(trans_x, trans_y, marker=markers[i % len(markers)], color=colors[i % len(colors)], label=method_name)
    # plt.title('Valid Percentage vs Translation Error Threshold', fontproperties=font)
    plt.xlabel('Translation Error Threshold (m)', fontproperties=font)
    plt.ylabel('Success Rate (%)', fontproperties=font)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Save the plot to a PNG file and pdf file
    plt.savefig(output_file_path + '/Translation_Error_Success_Rate_{}_{}.svg'.format(dataset_name, model_name), format="svg")
    plt.savefig(output_file_path + '/Translation_Error_Success_Rate_{}_{}.png'.format(dataset_name, model_name), bbox_inches='tight', dpi=500, pad_inches=0.1)
    plt.savefig(output_file_path + '/Translation_Error_Success_Rate_{}_{}.eps'.format(dataset_name, model_name), format="eps")
    pp = PdfPages(output_file_path + '/Translation_Error_Success_Rate_{}_{}.pdf'.format(dataset_name, model_name))
    pp.savefig()
    pp.close()
    plt.close()


if __name__ == '__main__':
    base_path = home_path + '/bash_scripts/output_pose_estimate'
    output_file_path = base_path + '/paper'
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    # folder_names = ["output_2501_2706", "output_2501_2706_raw3dgs", 
    #                 "output_2913_3233", "output_2913_3233_raw3dgs", 
    #                 "output_3972_4258"]
    # dataset_names = ["KITTI360_00_2501_2706", "KITTI360_00_2501_2706", 
    #                  "KITTI360_00_2913_3233", "KITTI360_00_2913_3233", 
    #                  "KITTI360_09_3972_4258"]
    # model_names = ["vegs", "r3dgs", "vegs", "r3dgs", "vegs"]

    folder_names = ["output_2913_3233_20_0.5", "output_2913_3233_r_20_0.5",
                    "output_2972_4258_20_0.5", "output_3972_4258_R_20_0.5", 
                    "output_3972_4258_10_10_1_1", "output_3972_4258_r_10_10_1_1",
                    "output_cz_10_1"]
    dataset_names = ["KITTI360_00_2913_3233_20_0.5", "KITTI360_00_2913_3233_r_20_0.5", 
                     "KITTI360_00_3972_4258_20_0.5", "KITTI360_00_3972_4258_r_20_0.5",
                     "KITTI360_00_3972_4258_10_10_1_1", "KITTI360_00_3972_4258_r_10_10_1_1",
                     "Wuhan_cz_10_1"]
    model_names = ["vegs", "vegs", "vegs", "vegs", "vegs", "vegs", "vegs"]

    for folder_name, dataset_name, model_name in zip(folder_names, dataset_names, model_names):
        plot_pose_estimation_results(os.path.join(base_path, folder_name), output_file_path, dataset_name, model_name)