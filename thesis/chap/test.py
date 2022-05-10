import os, sys, re

# Return all files in (sub)directory matching *.<pattern>
def get_all_files(dir, patterns):

    files = os.listdir(dir)
    all_files = list()

    for entry in files:
        # print(entry)
        full_path = os.path.join(dir, entry)
        if os.path.isdir(full_path):
            all_files = all_files + get_all_files(full_path, patterns)
        else:
            for pattern in patterns:
                if entry.endswith("." + pattern):
                    all_files.append(full_path)
                    break

    return all_files

# Return the directory of the LaTeX root file
def get_LaTeX_root_path(filename):

    with open(filename, "r") as tex_file:
        header = tex_file.readline()

    if header.startswith("%"):
        header = header[header.rfind("=") + 1:]
        header = header.replace(" ", "")
    else:
        header = filename

    root_file = os.path.abspath(header)

    return root_file[:root_file.rfind("/") + 1]

# Return all images files within root directory
def get_image_files(filename):

    img_extensions = ["png", "jpg", "jpeg", "JPG", "svg"]

    root_path = get_LaTeX_root_path(filename)
    images = get_all_files(root_path, img_extensions)
    rel_paths = [os.path.relpath(img, root_path) for img in images]

    return rel_paths

# Return all tex files within root directory
def get_tex_files(filename):

  root_path = get_LaTeX_root_path(filename)
  files = get_all_files(root_path,["tex"])

  for file in files:
      if "cover" in file or "include" in file:
          print(file)
          files.remove(file)

  print("\n")
  for file in files:
      print(file)

#   with open(filename) as f:
#     header = f.readline().strip()
#   if header.startswith("%"):
#     tex_files = [file.replace("/"+os.path.dirname(header[13:]),"") for file in tex_files]

#   return tex_files

get_tex_files(sys.argv[1])