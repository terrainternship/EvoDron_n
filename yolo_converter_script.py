import os
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import matplotlib.cm as cm
from matplotlib.patches import Patch
from shapely.geometry import MultiPolygon
from tqdm import tqdm
import sys
import numpy as np
from IPython.display import Audio
from shapely.errors import GEOSException
from collections import defaultdict

print("Запуск...")

draw_plots = draw_plots if 'draw_plots' in locals() else True
print(f"\n{YELLOW}Визуализация? {draw_plots}{RESET}")
translate_classes = translate_classes if 'translate_classes' in locals() else False
print(f"\n{YELLOW}Первод классов? {translate_classes}{RESET}")


# Get the list of files in the source folder
source_files = os.listdir(source_folder)
not_found_files = []

# Get the list of files in the source labels folder
source_labels_files = os.listdir(source_labels_folder)

# Check if the number of files in both folders is the same
if len(source_files) != len(source_labels_files):
    print(f"{YELLOW}Неравное количество файлов в папках: {len(source_files)} изображения к {len(source_labels_files)} аннотациям{RESET}")

elif len(source_files) == 0:
    print(f"{RED}Нет изображений в папке{RESET}")
    sys.exit()

class_id_to_name = {v: k for k, v in global_class_map.items()}

# Get a list of all the image files in the source folder
source_images = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]

# Initialize progress bar for source images
pbar_images = tqdm(total=len(source_images), desc="Обработка исходных изображений...")



total_sets_complete = 0

# Initialize a dictionary to store the areas of each class
class_areas = defaultdict(float)

# Define the groups of classes
group1 = {1, 2, 3}
group2 = {0, 4, 6, 7}

# Loop over all the image files
for source_image in source_images:

  # Define the paths to the image and its corresponding annotation file
  original_image_path = os.path.join(source_folder, source_image)
  original_annotation_path = os.path.join(source_labels_folder, source_image.replace('.jpg', '.txt'))

  try:
    # Load the original image
    original_image = Image.open(original_image_path)
  except FileNotFoundError:
      print(f"Image file {original_image_path} not found. Skipping to next image.")
      not_found_files.append(original_image_path)
      continue

  width, height = original_image.size

  # Store
  original_size = original_image.size
  original_width, original_height = width, height

  # Resize the image using the `resize` function
  crop_size = input_crop_size

  # Adjust scaling of labels if needed
  crop_adjustment = 1



  resized_image, over_12_chunks, chunks_to_make = resize_and_count_chunks(original_image, crop_size)
  width, height = resized_image.size


  if over_12_chunks:
      # Double the crop_size
      is_double_crop = True
      crop_size = (crop_size[0] * 2, crop_size[1] * 2)
      #crop_size = (crop_size[0] * 2, crop_size[1] * 2)
      print(f"Изображение займет больше чем 12 640х640 кусков, удваиваем размер куска {crop_size}...")
  else:
      is_double_crop = False
      crop_size = crop_size

  ############## if plots enabled
  if draw_plots:
  # Create a figure and axis to display the image
    fig, ax = plt.subplots(1)
    ax.imshow(resized_image)
    ax.set_title(f'Исходное изображение {resized_image.size}\n{source_image}' if not is_double_crop else f'Исходное из. {original_image.size}, размер изменен на {resized_image.size}\n{source_image}', fontsize=10)


    print("Рисую исходное изображение и разметку...")

  # Load and display the original annotations as polygons
  try:
      with open(original_annotation_path, 'r') as annotation_file:
        for line in annotation_file:
            parts = line.strip().split()

            source_class_id = int(parts[0])

            if translate_classes:
                source_class_id = convert_number(source_class_id)


            used_classes.add(source_class_id)

            ############## if plots enabled
            if draw_plots:
                vertices = [float(x) for x in parts[1:]]

                # Convert normalized coordinates to pixel coordinates
                pixel_vertices = [(int(vertices[i] * width), int(vertices[i + 1] * height)) for i in range(0, len(vertices), 2)]

                polygon = plt.Polygon(
                    pixel_vertices,
                    fill=False,
                    edgecolor=class_colors_mpl_normalized[source_class_id],
                    linewidth=2
                )

                # Add the polygon to the plot
                ax.add_patch(polygon)

                # Add the label to the legend_labels list
                #legend_labels.append(str(source_class_id_translation))

  except FileNotFoundError:
    print(f"Annotation file {original_annotation_path} not found. Skipping to next image.")
    not_found_files.append(original_annotation_path)
    continue

  ############## if plots enabled
  if draw_plots:

      # Use legend_labels for creating the legend
      legend_patches = [
      Patch(color=class_colors_mpl_normalized[class_index], label=str(class_index))
      for class_index in used_classes
      ]

      ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


      # Show the image with annotations
      plt.show()

  '''  # Calculate the padding needed to make the image size a multiple of the crop size
  pad_width = crop_size[0] - width % crop_size[0] if width % crop_size[0] != 0 else 0
  pad_height = crop_size[1] - height % crop_size[1] if height % crop_size[1] != 0 else 0

  # Create a new image with the padded size and a checkerboard pattern
  padded_image = Image.new('RGB', (width + pad_width, height + pad_height))

  # Create a draw object
  draw = ImageDraw.Draw(padded_image)

  # Define the size of the squares
  square_size = int(max(width, height) * 0.01)  # Adjust this value to change the size of the squares

  # Draw the checkerboard pattern
  for i in range(0, width + pad_width, square_size):
      for j in range(0, height + pad_height, square_size):
          if (i // square_size) % 2 == (j // square_size) % 2:
              draw.rectangle([i, j, i + square_size, j + square_size], fill='white')
          else:
              draw.rectangle([i, j, i + square_size, j + square_size], fill='gray')

  # Paste the original image onto the padded image
  padded_image.paste(original_image, (0, 0))

  # Replace the original image with the padded image
  original_image = padded_image

  # Store the original image dimensions
  original_width, original_height = width, height

  # Pad the original image
  original_image = ImageOps.expand(original_image, (0, 0, pad_width, pad_height)) '''

  # Update the image size
  #width, height = resized_image.size

  '''   # Calculate the scale factors for width and height
  scale_width = original_width / width
  scale_height = original_height / height '''


  # Calculate the number of rows and columns for cropping
  num_rows = max(1, height // crop_size[1])
  num_cols = max(1, width // crop_size[0])


  ############## if plots enabled
  # Create a subplot grid
  if draw_plots:
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 8))
    print("\nПодготавливаю изображения и разметку...")


  # Loop through each row and column to crop and save images
  for row in range(num_rows):
      for col in range(num_cols):

          # Calculate the cropping region
          left = int(col * crop_size[0])
          top = int(row * crop_size[1])
          right = int(left + crop_size[0])
          bottom = int(top + crop_size[1])

          # Calculate the region for the current cropped image
          region = (left / width, top / height, right / width, bottom / height)

          # Crop the image
          cropped_image = resized_image.crop((left, top, right, bottom))

          # Load original annotations
          with open(original_annotation_path, 'r') as annotation_file:
              annotations = []
              for line in annotation_file:
                  parts = line.strip().split()
                  source_class_id = (int(parts[0]))

                  # Check if the source class ID exists in the translation dictionary
                  if translate_classes:
                      source_class_id = convert_number(source_class_id)

                  # Now you can use target_class_id in your code as needed
                  source_class_name = class_id_to_name[int(source_class_id)]

                  # Parse normalized coordinates from the original annotation
                  coordinates = list(map(float, parts[1:]))

                  # Scale the coordinates of the polygons
                  scaled_coordinates = [(coordinates[i] * 1, coordinates[i + 1] * 1) for i in range(0, len(coordinates), 2)]

                  # Create a polygon from the vertices
                  polygon = Polygon(scaled_coordinates)

                  # Get the width and height of the bounding box
                  bbox_width = float(parts[3]) * width
                  bbox_height = float(parts[4]) * height

                  # Calculate the area of the bounding box
                  bbox_area = bbox_width * bbox_height

                  # Add the area of the bounding box to the total area of the class
                  class_areas[source_class_id] += bbox_area

                  # Create a box representing the boundary of the cropped region
                  boundary = box(region[0], region[1], region[2], region[3])

                  try:
                      clipped_polygon = polygon.intersection(boundary)
                  except GEOSException as e:

                      print(f"GEOSException encountered: {e}")

                      continue  # Skip this iteration or add custom handling

                  # Check if the clipped polygon is not empty
                  if not clipped_polygon.is_empty:

                      # Check if the clipped polygon is a MultiPolygon
                      if isinstance(clipped_polygon, MultiPolygon):

                        # Handle each Polygon in the MultiPolygon separately
                        for poly in clipped_polygon.geoms:

                            # Convert the clipped polygon to a list of vertices
                            clipped_vertices = list(poly.exterior.coords)


                            # Translate normalized coordinates to cropped image coordinates
                            clipped_vertices = [((x - left) / crop_size[0], (y - top) / crop_size[1]) for x, y in clipped_vertices]


                      else:

                          # Convert the clipped polygon to a list of vertices
                          if isinstance(clipped_polygon, Polygon):
                            clipped_vertices = list(clipped_polygon.exterior.coords)

                          # Translate normalized coordinates to cropped image coordinates
                          clipped_vertices = [((x - region[0]) / (region[2] - region[0]), (y - region[1]) / (region[3] - region[1])) for x, y in clipped_vertices]

                          # Append the annotation to the list for this cropped image
                          annotations.append(
                              f'{source_class_id} ' + ' '.join(map(str, [coord for vertex in clipped_vertices for coord in vertex]))
                          )

              # Resize the cropped image back to target crop size if it was doubled
              if is_double_crop:
                cropped_image=resize_from_double(cropped_image)
                crop_adjustment = 2

              # Save the cropped image
              output_image_path = os.path.join(output_img_folder, f'{source_image}_output_image_{row}_{col}.jpg')
              cropped_image.save(output_image_path)


              ############## if plots enabled
              if draw_plots:
                # Create a draw object
                draw = ImageDraw.Draw(cropped_image)

                # Loop through annotations for the current output image
                for annotation in annotations:
                    parts = annotation.split()

                    # Get the class label
                    class_label = parts[0]


                    # Parse normalized coordinates from the annotation
                    coordinates = list(map(float, parts[1:]))
                    vertices = []
                    for i in range(0, len(coordinates), 2):
                          x = coordinates[i] * crop_size[0] / crop_adjustment
                          y = coordinates[i + 1] * crop_size[1] / crop_adjustment

                          # Adjust positions via shift values if needed

                          ''' # Calculate the shift value
                          shift_value_x = 0 if pad_width == 0 or x == crop_size[0] else 0.01 * pad_width + 0.02 * (x + ((col)* crop_size[0]))
                          shift_value_y = 0 if pad_height == 0 or y == crop_size[1] else  0.02 * (y + ((row)* crop_size[1]))

                          print(f"pw: {pad_width}, ph: {pad_height} w:{width} col: {col}, sX: {shift_value_x}, sY: {shift_value_y}, x:{x}, y:{y}") '''

                          shift_value_x = 0
                          shift_value_y = 0
                          vertices.append((x - shift_value_x, y - shift_value_y))

                    # Draw the polygon with the color corresponding to the class
                    draw.polygon(vertices, outline=class_colors_mpl[class_id_to_name[int(class_label)]], width=8)


                # Display the cropped image with annotations
                fig.suptitle(f'Полученные изображения @ {crop_size[0]} x {crop_size[1]}\n{source_image}' if not is_double_crop else f'Полученные изображения @ {crop_size[0]} x {crop_size[1]}\n{source_image}, сохранены уменьшенными вдвое', fontsize=10)
                if num_rows == 1:
                    if num_cols == 1:
                        axs.imshow(cropped_image)
                        axs.set_title(f'Output Image {row}_{col}')
                        axs.set_aspect('equal')
                    else:
                        axs[col].imshow(cropped_image)
                        axs[col].set_title(f'Output Image {row}_{col}')
                        axs[col].set_aspect('equal')
                elif num_cols == 1:
                    axs[row].imshow(cropped_image)
                    axs[row].set_title(f'Output Image {row}_{col}')
                    axs[row].set_aspect('equal')
                else:
                    axs[row, col].imshow(cropped_image)
                    axs[row, col].set_title(f'Output Image {row}_{col}')
                    axs[row, col].set_aspect('equal')

          total_sets_complete += 1

          # Save annotations to a file with the same name as the image
          output_annotation_path = os.path.join(output_labels_folder, f'{source_image}_output_image_{row}_{col}.txt')
          with open(output_annotation_path, 'w') as annotation_output_file:
              for annotation in annotations:

                  parts = annotation.split()
                  source_class_id = int(parts[0])
                  global_class_id = source_class_id
                  parts[0] = str(global_class_id)
                  annotation_output_file.write(' '.join(parts) + '\n')


  pbar_images.update()
  sys.stdout.flush()


  ############## if plots enabled
  if draw_plots:

    # Remove empty subplots (if the number of images is not a multiple of images_per_row)
    for i in range(num_rows * num_cols, num_rows):
        axs[i].axis('off')

  ############## if plots enabled
  if draw_plots:
    fig.legend(handles=legend_patches, loc='upper right')
    plt.tight_layout()
    plt.show()


# Close progress bar for source images
pbar_images.close()




print(f"\n{GREEN}{total_sets_complete} sets of images & labels ready, run the cells below to download ZIP{RESET}")
print("*********")
print(f"{GREEN}{total_sets_complete} Изображений+лейблов сделаны. Запустите ячейки ниже, чтобы скачать ZIP.{RESET}")

# Print the total area of each class
for class_id, area in class_areas.items():
    print(f"Class {class_id}: {area} pixels")

# Calculate the total area of each group
group1_total_area = sum(class_areas[class_id] for class_id in group1)
group2_total_area = sum(class_areas[class_id] for class_id in group2)

# Calculate the balance of each group
if group1_total_area + group2_total_area > 0:
    group1_balance = group1_total_area / (group1_total_area + group2_total_area)
    group2_balance = group2_total_area / (group1_total_area + group2_total_area)
else:
    group1_balance = group2_balance = 0

# Print the balance of each group
print(f"Group 1: {group1_balance * 100}%")
print(f"Group 2: {group2_balance * 100}%")

# Plot the balance of each group
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
if group1_total_area > 0:
    plt.bar(['Class 1', 'Class 2', 'Class 3'], [class_areas[1]/group1_total_area*100, class_areas[2]/group1_total_area*100, class_areas[3]/group1_total_area*100])
else:
    plt.bar(['Class 1', 'Class 2', 'Class 3'], [0, 0, 0])
plt.title('Group 1 Class Proportions')
plt.xlabel('Class')
plt.ylabel('Proportion of Total Area (%)')

plt.subplot(1, 2, 2)
if group2_total_area > 0:
    plt.bar(['Class 0', 'Class 4', 'Class 6', 'Class 7'], [class_areas[0]/group2_total_area*100, class_areas[4]/group2_total_area*100, class_areas[6]/group2_total_area*100, class_areas[7]/group2_total_area*100])
else:
    plt.bar(['Class 0', 'Class 4', 'Class 6', 'Class 7'], [0, 0, 0, 0])
plt.title('Group 2 Class Proportions')
plt.xlabel('Class')
plt.ylabel('Proportion of Total Area (%)')

plt.tight_layout()
plt.show()

if not_found_files:
    print(f"These files were not found ({len(not_found_files)}):")
    for file in not_found_files:
        print(f'"{file}"')