import json
import os
import sys

import matplotlib.pyplot as plt

home_path = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
print(home_path)
if home_path not in sys.path:
    sys.path.append(home_path)

import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages

# setting font globally
font_path = home_path + '/visualization/fonts/Times New Roman.ttf'
font = fm.FontProperties(fname=font_path, size=12)


ITERATION = 'ours_30000' #ours_100000

def plot_psnr(file_paths, method_names, output_file_path, dataset_name, model_name):
    # Prepare the plot
    plt.figure(figsize=(15, 10))

    # Initialize subplots for SSIM, PSNR, and LPIPS
    for idx, metric in enumerate(['SSIM', 'PSNR', 'LPIPS']):
        plt.subplot(3, 1, idx + 1)
        for file_path, method_name in zip(file_paths, method_names):
            # Load the JSON data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract and sort the image sequences
            images = sorted(data[ITERATION][metric].keys())
            image_ids = list(range(len(images)))
            values = [data[ITERATION][metric][img] for img in images]
            
            # Plot the metric for the current method
            plt.plot(image_ids, values, label=method_name, marker="o")
        
        # Add titles, labels, and grid to each subplot
        plt.title(f"{metric} over test image set", fontproperties=font)
        plt.xlabel("Image ID", fontproperties=font)
        plt.ylabel(metric, fontproperties=font)
        plt.grid(True)
        plt.legend()  # Show the legend for each subplot

    # Adjust layout
    plt.tight_layout()

    # Save the plot to a PNG file and pdf file
    plt.savefig(output_file_path + '/comparison_metrics_{}_{}.svg'.format(dataset_name, model_name), format="svg")
    plt.savefig(output_file_path + '/comparison_metrics_{}_{}.png'.format(dataset_name, model_name), bbox_inches='tight', dpi=500, pad_inches=0.1)
    plt.savefig(output_file_path + '/comparison_metrics_{}_{}.eps'.format(dataset_name, model_name), format="eps")
    pp = PdfPages(output_file_path + '/comparison_metrics_{}_{}.pdf'.format(dataset_name, model_name))
    pp.savefig()
    pp.close()
    plt.close()

# Optionally display the plot
# plt.show()

# # Loop through each metric and create a separate plot for each
# for metric in ['SSIM', 'PSNR', 'LPIPS']:
#     plt.figure(figsize=(15, 10))  # Create a new figure for each metric
#     for file_path, method_name in zip(file_paths, method_names):
#         # Load the JSON data
#         with open(file_path, 'r') as f:
#             data = json.load(f)
        
#         # Extract and sort the image sequences
#         images = sorted(data['ours_100000'][metric].keys())
#         image_ids = list(range(len(images)))
#         values = [data['ours_100000'][metric][img] for img in images]
        
#         # Plot the metric for the current method
#         plt.plot(image_ids, values, label=method_name, marker="o")
    
#     # Add titles, labels, and grid to the plot
#     plt.title(f"{metric} over test image set")
#     plt.xlabel("Image ID")
#     plt.ylabel(metric)
#     plt.grid(True)
#     plt.legend()  # Show the legend for the plot

#     # Save each plot to a separate PNG file with 500 DPI and tight layout
#     plt.savefig(f'{metric}_comparison_500dpi.png', dpi=500, bbox_inches='tight')
#     plt.close()  # Close the current plot before starting the next one

if __name__ == '__main__':
    output_file_path = home_path + '/bash_scripts/paper'
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    dataset_name = "MLS" # KITTI360_09
    model_name = "GSLoc"

    # List of file paths and corresponding method names
    # file_paths = [output_file_path + '/per_view_results_reorganized/2501_3233_raw3dgs.json', 
    #             output_file_path + '/per_view_results_reorganized/2501_3233_lidar.json',
    #             output_file_path + '/per_view_results_reorganized/2501_3233_vegs.json']
    # file_paths = [output_file_path + '/per_view_results_reorganized/3972_4258_raw3dgs.json', 
    #               output_file_path + '/per_view_results_reorganized/3972_4258_lidar.json',
    #               output_file_path + '/per_view_results_reorganized/3972_4258_vegs.json']
    # method_names = ['3DGS(SfM)', '3DGS(LiDAR)', '3DGS(SfM+LiDAR)']
    
    file_paths = [output_file_path + '/per_view_results_wuhan/per_view_test_cz_cz_03.json']
    method_names = ['3DGS(LiDAR)']
    
    
    plot_psnr(file_paths, method_names, output_file_path, dataset_name, model_name)