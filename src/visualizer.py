"""
Visualization module for extracted profile parameters.

Generates a 2D plot of the profile with overlaid dimension annotations
using matplotlib.
"""

from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from build123d import Vertex
from .geometry import ProfileParams

def plot_profile_dims(
    vertices: List[Vertex], 
    params: ProfileParams, 
    output_path: Path
) -> None:
    """
    Generate and save a 2D Dimensioned Drawing image.
    
    Args:
        vertices: List of 16 vertices (P1..P15..P0 order typically from step_analyzer)
                  Wait, step_analyzer returns P1..P15..P0.
        params: Extracted parameters to label
        output_path: Path to save the image
    """
    
    # Extract (Y, Z) coordinates for plotting
    # In our profile: Y is horizontal, Z is vertical
    coords = [(v.Y, v.Z) for v in vertices]
    
    # Close the loop
    coords.append(coords[0])
    
    ys, zs = zip(*coords)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(ys, zs, 'b-', linewidth=2, label='Profile')
    ax.scatter(ys, zs, color='red', s=20, zorder=5)
    
    # Helper to draw dimension line
    def draw_dim(p1_idx, p2_idx, label, offset=0.1, text_offset=0.02, axis='y'):
        if p1_idx >= len(vertices) or p2_idx >= len(vertices):
            return

        p1 = vertices[p1_idx]
        p2 = vertices[p2_idx]
        
        y1, z1 = p1.Y, p1.Z
        y2, z2 = p2.Y, p2.Z
        
        # Center point
        mid_y = (y1 + y2) / 2
        mid_z = (z1 + z2) / 2
        
        # Annotation properties
        prop = dict(arrowstyle='<->', shrinkA=0, shrinkB=0, linewidth=1.0, color='black')
        
        if axis == 'y':
            # Horizontal dimension
            dim_z = mid_z + offset # Shift vertically
            ax.annotate('', xy=(y1, dim_z), xytext=(y2, dim_z), arrowprops=prop)
            # Projection lines
            ax.plot([y1, y1], [z1, dim_z], 'k:', linewidth=0.5)
            ax.plot([y2, y2], [z2, dim_z], 'k:', linewidth=0.5)
            # Text
            ax.text(mid_y, dim_z + text_offset, f"{label}={params.__getattribute__(label)}", 
                    ha='center', va='bottom', fontsize=9, color='darkblue')
            
        elif axis == 'z':
            # Vertical dimension
            dim_y = mid_y + offset # Shift horizontally
            ax.annotate('', xy=(dim_y, z1), xytext=(dim_y, z2), arrowprops=prop)
            # Projection lines
            ax.plot([y1, dim_y], [z1, z1], 'k:', linewidth=0.5)
            ax.plot([y2, dim_y], [z2, z2], 'k:', linewidth=0.5)
            # Text
            ax.text(dim_y + text_offset, mid_z, f"{label}={params.__getattribute__(label)}", 
                    ha='left', va='center', fontsize=9, color='darkblue', rotation=90)

    # Assuming vertices are ordered P1, P2, ... P15, P0
    # Map indices:
    # 0:P1, 1:P2, 2:P3, 3:P4, 4:P5, 5:P6, 6:P7, 7:P8, 8:P9, 9:P10, 10:P11, 11:P12, 12:P13, 13:P14, 14:P15, 15:P0
    
    # L3: P1-P2 (Vertical) -> idx 0-1
    draw_dim(0, 1, 'L3', offset=-0.1, axis='z') # Right side offset
    
    # L2: P0-P1 (Horizontal) -> idx 15-0
    draw_dim(15, 0, 'L2', offset=0.15, axis='y') # Top offset
    
    # L1: P14-P15 (Horizontal Y) -> idx 13-14
    draw_dim(13, 14, 'L1', offset=0.1, axis='y')

    # L4: P4-P3 (Y) -> idx 3-2
    draw_dim(3, 2, 'L4', offset=0.05, axis='y')
    
    # L6: P5-P4 (Z) -> idx 4-3
    draw_dim(4, 3, 'L6', offset=0.1, axis='z')
    
    # L5: P5-P6 (Y) -> idx 4-5
    draw_dim(4, 5, 'L5', offset=-0.1, axis='y') # Bottom side of top block
    
    # L7: P6-P7 (Z) -> idx 5-6
    draw_dim(5, 6, 'L7', offset=-0.05, axis='z')
    
    # L8: P9-P8 (Y) -> idx 8-7
    draw_dim(8, 7, 'L8', offset=0.05, axis='y')
    
    # L9: P9-P10 (Z) -> idx 8-9
    draw_dim(8, 9, 'L9', offset=-0.05, axis='z')
    
    # R1: P10-P11 corner
    # Just label the point
    ax.annotate(f"R1={params.R1}", xy=(ys[10], zs[10]), xytext=(ys[10]+0.2, zs[10]+0.2),
                arrowprops=dict(arrowstyle='->', color='green'), color='green')
                
    # R2: P12-P13 corner
    ax.annotate(f"R2={params.R2}", xy=(ys[12], zs[12]), xytext=(ys[12]+0.2, zs[12]-0.2),
                arrowprops=dict(arrowstyle='->', color='green'), color='green')
    
    # L10: P14-P13 (Z) -> idx 13-12
    draw_dim(13, 12, 'L10', offset=-0.15, axis='z')

    ax.set_aspect('equal')
    ax.set_title(f"Extracted Dimensions: {output_path.stem}")
    ax.axis('off')
    
    # Determine bounds for better scaling
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
