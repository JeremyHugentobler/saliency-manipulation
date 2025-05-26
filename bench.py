from pathlib import Path
from argparse import ArgumentParser
from PIL import Image


REF_PATH = Path(".\data\saliency_shift")
MASK_PATH = REF_PATH / "masks"
ORIGINAL_PREFIX = "_in"
OUT = "html"



def placeholder(img_path, mask_path):
    """
    Placeholder function to be replaced with the actual implementation.
    
    Returns:
        PIL.Image: The image with the saliency shift applied.
    """
    return Image.open(img_path) # Identity function for now


def render_html(image_to_mask, output_path, all_images):
    """
    Render the results in HTML format with all images of the same ID in a single row.
    """
    
    html = "<html><head><title>Saliency Shift Benchmark</title></head><body>"
    
    # Updated CSS for better horizontal comparison
    html += """
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            padding: 20px;
            background-color: #f4f4f4;
        }
        h1, h2 {
            color: #333;
        }
        img {
            max-width: 100%;
            height: auto;
            margin: 10px 0;
        }
        .image-row {
            display: flex;
            flex-direction: row;
            flex-wrap: nowrap;
            overflow-x: auto;
            margin-bottom: 30px;
            background-color: #fff;
            border-radius: 0 0 5px 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
        }
        .image-reference {
            display: flex;
            flex-direction: row;
            justify-content: center;
            align-items: center;
            overflow-x: auto;
            flex-wrap: nowrap;
            background-color: #fff;
            border-radius: 5px 5px 0 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
        }
        .image-container {
            flex: 0 0 auto;
            margin: 10px;
            padding: 10px;
            text-align: center;
            min-width: 250px;
        }
        .image-container img {
            max-width: 70%;
            height: auto;
            max-height: 300px; /* Limit height for consistency */
        }
        .image-container h3 {
            font-size: 1em;
            margin: 10px 0;
            color: #333;
        }
        .image-title {
            width: 100%;
            text-align: left;
            margin-bottom: 10px;
            padding-left: 20px;
        }
    </style>
    """        
    
    html += "<h1>Saliency Results Comparison</h1>"
    
    # Group all images by ID for side-by-side comparison
    for img_path, mask_path in image_to_mask.items():
        # Get the image name and ID
        img_name = img_path.stem
        mask_name = mask_path.stem
        id = img_name.replace("_in", "")
        ours = img_path.stem.replace("_in", "_ours")
        paper = img_path.stem.replace("_in", "_out")
                
        # Start a new row for this image set
        html += f"""
        <h2 class="image-title">Image Set: {id}</h2>
        <div class="image-reference">
            <div class="image-container">
                <h3>Original Image</h3>
                <img src="{img_path}" alt="Original image">
            </div>
            <div class="image-container">
                <h3>Mask</h3>
                <img src="{mask_path}" alt="Mask image">
            </div>
        </div>
        <div class="image-row">
            <div class="image-container">
                <h3>Our implementation</h3>
                <img src="{REF_PATH / (ours + ".jpg")}" alt="Ours image">
            </div> 
            <div class="image-container">
                <h3>Paper implementation</h3>
                <img src="{REF_PATH / (paper + ".jpg")}" alt="Paper image">
            </div>
        
        """
        
        # Add all related images in the same row
        for a_img_path in all_images:
            if id in a_img_path.stem and a_img_path != img_path and a_img_path != mask_path and a_img_path.stem not in [ours, paper]:
                method_name = a_img_path.stem.replace(id, "").strip("_")
                html += f"""
                <div class="image-container">
                    <h3>{method_name}</h3>
                    <img src="{a_img_path}" alt="{a_img_path.stem}">
                </div>
                """
        
        # Close the row
        html += "</div>"

    html += "</body></html>"
    with open(output_path, "w") as f:
        f.write(html)
    print(f"- Results saved to {output_path}")
    
    
def render_latex(image_to_mask, output_path, all_images):
    """
    Render the results in LaTeX format with all images of the same ID in a row layout.
    """
    latex = r"""
    \documentclass{article}
    \usepackage{graphicx}
    \usepackage{float}
    \usepackage{caption}
    \usepackage{subcaption}
    \usepackage{geometry}
    \geometry{margin=1in}
    
    % For better image handling
    \usepackage{grffile}
    
    % For equal height subfigures
    \usepackage{adjustbox}
    
    \begin{document}
    \title{Saliency Shift Benchmark}
    \author{David, George, Jérémy}
    \maketitle
    \section{Results}
    """
    
    for img_path, mask_path in image_to_mask.items():
        
        img_name = img_path.stem
        mask_name = mask_path.stem
        id = img_name.replace("_in", "")
        ours = img_path.stem.replace("_in", "_ours")
        paper = img_path.stem.replace("_in", "_out")
        
        # Reference and mask - First row
        latex += f"""
        \\subsection{{Image Set: {id}}}
        
        % Original and Mask in first row
        \\begin{{figure}}[H]
            \\centering
            \\begin{{subfigure}}[t]{{0.45\\textwidth}}
                \\centering
                \\includegraphics[width=\\textwidth]{{{img_path}}}
                \\caption{{Original Image}}
                \\label{{fig:{img_name}}}
            \\end{{subfigure}}
            \\hfill
            \\begin{{subfigure}}[t]{{0.45\\textwidth}}
                \\centering
                \\includegraphics[width=\\textwidth]{{{mask_path}}}
                \\caption{{Mask}}
                \\label{{fig:{mask_name}}}
            \\end{{subfigure}}
            \\caption{{Reference images for {id}}}
        \\end{{figure}}
        
        % Results row with all methods
        \\begin{{figure}}[H]
            \\centering
        """
        
        # Calculate how many images per row (standard is 3)
        related_images = [img for img in all_images if id in img.stem 
                         and img != img_path and img != mask_path]
        # Add our implementation and paper implementation
        default_methods = 2  # Our implementation and paper implementation
        total_methods = len(related_images) + default_methods
        
        # Add our implementation
        latex += f"""
            \\begin{{subfigure}}[t]{{0.3\\textwidth}}
                \\centering
                \\includegraphics[width=\\textwidth]{{{REF_PATH / (ours + ".jpg")}}}
                \\caption{{Our implementation}}
                \\label{{fig:{ours}}}
            \\end{{subfigure}}
            \\hfill
        """
        
        # Add paper implementation
        latex += f"""
            \\begin{{subfigure}}[t]{{0.3\\textwidth}}
                \\centering
                \\includegraphics[width=\\textwidth]{{{REF_PATH / (paper + ".jpg")}}}
                \\caption{{Paper implementation}}
                \\label{{fig:{paper}}}
            \\end{{subfigure}}
        """
        
        # Add any additional methods that may exist
        for idx, a_img_path in enumerate(related_images):
            if id in a_img_path.stem and a_img_path != img_path and a_img_path != mask_path and a_img_path.stem not in [ours, paper]:
                method_name = a_img_path.stem.replace(id, "").strip("_")
                
                # Add hfill between images but not after the last one
                separator = "\\hfill" if idx < len(related_images) - 1 else ""
                
                latex += f"""
                \\hfill
                \\begin{{subfigure}}[t]{{0.3\\textwidth}}
                    \\centering
                    \\includegraphics[width=\\textwidth]{{{a_img_path}}}
                    \\caption{{{method_name}}}
                    \\label{{fig:{a_img_path.stem}}}
                \\end{{subfigure}}
                {separator}
                """
        
        # Close the figure
        latex += f"""
            \\caption{{Different saliency implementations for {id}}}
        \\end{{figure}}
        """
        
    latex += r"""
    \end{document}
    """
    
    # Save the LaTeX file
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"- Results saved to {output_path}")
    
    
# Entry point
if __name__ == "__main__":

    # Check if there are any arguments
    parser = ArgumentParser(description="Benchmarking script for saliency shift.")
    # Path to the images
    parser.add_argument(
        "--dataset",
        type=str,
        default=REF_PATH,
        help="Dataset to use for benchmarking.",
    )
    
    # Prefix for the original images
    parser.add_argument(
        "--original_prefix",
        type=str,
        default=ORIGINAL_PREFIX,
        help="Prefix for the original images. ('in' by default)",
    )        
    
    # The kind of output (html or latex)
    parser.add_argument(
        "--out",
        type=str,
        default=OUT,
        choices=["html", "latex"],
        metavar="OUTPUT",
        help="Output format. ('html' or 'latex')",
    )
    
    # Adapt the constants
    args = parser.parse_args()
    REF_PATH = Path(args.dataset)
    MASK_PATH = REF_PATH / "masks"
    ORIGINAL_PREFIX = args.original_prefix
    # Check if the paths exist
    if not REF_PATH.exists():
        raise ValueError(f"Path {REF_PATH} does not exist.")
    if not MASK_PATH.exists():
        raise ValueError(f"Path {MASK_PATH} does not exist.")
    # Check if the paths are directories
    if not REF_PATH.is_dir():
        raise ValueError(f"Path {REF_PATH} is not a directory.")
    if not MASK_PATH.is_dir():
        raise ValueError(f"Path {MASK_PATH} is not a directory.")
    
    # Get the list of images
    extensions = [".jpg", ".png", ".jpeg"]
    images_path = []
    in_images_path = []
    for ext in extensions:
        # get all the images, but exclude subdirectories
        
        images_path.extend(REF_PATH.glob(f"**/*{ext}"))
        in_images_path.extend(REF_PATH.glob(f"**/*{ORIGINAL_PREFIX}*{ext}"))
    # Remove the subdirectories
    images_path = [img for img in images_path if img.parent == REF_PATH]
    in_images_path = [img for img in in_images_path if img.parent == REF_PATH]
        
    # Get the masks
    masks_path = []
    for ext in extensions:
        masks_path.extend(MASK_PATH.glob(f"**/*{ext}"))
        
    print(f"- Found {len(in_images_path)} input images in {REF_PATH}.")
    
    image_to_mask = {}
    
    for img_path in in_images_path:
        # Get the corresponding mask
        mask_path = None
        out = img_path.stem.replace(ORIGINAL_PREFIX, "")
        for m_path in masks_path:
            if m_path.stem == out + "_mask":
                image_to_mask[img_path] = m_path
                mask_path = m_path
                break
        if mask_path is None:
            raise ValueError(f"Mask for {img_path} not found.")
        
        # Apply the saliency shift
        img = placeholder(img_path, mask_path)
        
        # Save the image
        suffix = "_ours.jpg"
        out = out + suffix
        
        # If it's not a PIL image, convert it
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img.save(REF_PATH / out)
        
        
    
    # Output
    OUT = args.out
    if OUT == "html":
        print("- Output format: HTML")
        render_html(image_to_mask, "results.html", images_path)
    elif OUT == "latex":
        print("- Output format: LaTeX")
        render_latex(image_to_mask, "results.tex", images_path)
    else:
        raise ValueError(f"Output format {OUT} not supported.")
