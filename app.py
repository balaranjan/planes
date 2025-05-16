import streamlit as st
from io import StringIO
import numpy as np
from cifkit import Cif
from matplotlib import pyplot as plt
from tempfile import NamedTemporaryFile
from math import ceil
from cifkit.utils import unit


def plane_equation_from_points(p1, p2, p3):

    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # Find two vectors in the plane
    v1 = p2 - p1
    v2 = p3 - p1

    normal = np.cross(v1, v2)
    a, b, c = normal
    d = np.dot(normal, p1)

    return (a, b, c, d)


def are_planes_same(plane1, plane2, tolerance=1e-6):

    a1, b1, c1, d1 = plane1
    a2, b2, c2, d2 = plane2
    
    normal1 = np.array([a1, b1, c1])
    normal2 = np.array([a2, b2, c2])
    cross_product = np.cross(normal1, normal2)
    
    if not np.allclose(cross_product, [0, 0, 0], atol=tolerance):
        return False
    
    # Find scaling factor k using the first non-zero component
    for i in range(3):
        denominator = normal2[i]
        if abs(denominator) > tolerance:
            k = normal1[i] / denominator
            break
    else:
        print("Invalid plane (zero normal)")
        return False  
    
    return (np.allclose(normal1, k * normal2, atol=tolerance) and 
            np.isclose(d1, k * d2, atol=tolerance))
    
    
def are_collinear_3d(p1, p2, p3, tolerance=1e-12):
    # Check if three 3D points are collinear using vector cross product.

    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    
    vec_ab = b - a
    vec_ac = c - a

    cross_product = np.cross(vec_ab, vec_ac)
    cross_magnitude = np.linalg.norm(cross_product)
    
    return cross_magnitude < tolerance


def num_atoms_on_plane(plane_coeff, points, threshold, nmax=None):
    # Calculate the distance from each point to the plane Ax + By + Cz + D = 0.

    # points = np.array([p[:-1] for p in points])
    if nmax is not None:
        points = np.vstack([p[-1] for p in points][:12])
    # print('shape', points, points.shape, len(points))

    A, B, C, D = plane_coeff
    numerator = np.abs(A * points[:, 0] + B * points[:, 1] + C * points[:, 2] + D)
    denominator = np.sqrt(A**2 + B**2 + C**2)
    
    if denominator == 0:
        raise ValueError("Invalid plane coefficients: A, B, C cannot all be zero.")
    
    distances =  numerator / denominator
    
    return (np.abs(distances) <= threshold).astype(int).sum(), points[(np.abs(distances) <= threshold)]


def get_unique_planes(site_conns, threshold=0.5, nmax=20):
    unique_planes = []
    site_planes = []
    sorted_distaces = []
    selected_planes = []
    
    for site, points_wd in site_conns.items():
        center = points_wd[0][-2]
        points_wd = sorted(points_wd, key=lambda x: x[1])[:nmax]
        s_planes = []
        for i in range(nmax):
            for j in range(i+1, nmax):
                for k in range(j+1, nmax):
                    p1, p2, p3 = points_wd[i][-1], points_wd[j][-1], points_wd[k][-1]
                    if are_collinear_3d(p1, p2, p3):
                        continue
                    
                    new_plane = plane_equation_from_points(p1, p2, p3)
                    
                    # center in plane
                    center_in_plane, _ = num_atoms_on_plane(new_plane, np.array([center]), threshold)
                    
                    if not center_in_plane:
                        continue
                    # s_planes.append(new_plane)
                    unique = True
                    if len(unique_planes):
                        for plane in unique_planes:
                            if are_planes_same(plane, new_plane):
                                # unique_planes.append(new_plane)
                                unique = False
                                break
                            
                            
                        for plane in unique_planes:
                            num_p, pp = num_atoms_on_plane(plane, [[p] for p in [p1, p2, p3]], threshold, nmax=12)
                            if num_p == 3:
                                unique = False
                    else:
                        unique_planes.append(new_plane) 
                        unique = True
                    
                    if unique:
                        unique_planes.append(new_plane)
                        num_points, points_on_plane = num_atoms_on_plane(new_plane, points_wd, threshold, nmax=12)
                        
                        distances = sorted(np.linalg.norm(points_on_plane - center, axis=1))
                        unique_dist = True
                        if len(sorted_distaces):
                            for sd in sorted_distaces:
                                if len(sd) != len(distances):
                                    continue
                                if np.allclose(sd, distances):
                                    unique_dist = False
                        else:
                            sorted_distaces.append(distances)
                            
                        
                        if unique_dist:

                            s_planes.append(new_plane)
                            sorted_distaces.append(distances)
                            selected_planes.append([new_plane, site, center, points_on_plane])

        site_planes.append([center, s_planes])
    return site_planes, selected_planes


def align_plane_to_xy(plane_coeffs, points):
    """
    Rotates a 3D plane and its points to align with the xy-plane.
    """
    A, B, C, D = plane_coeffs
    
    # Step 1: Translate plane to origin
    if C != 0:
        t = np.array([0, 0, -D/C])
    elif B != 0:
        t = np.array([0, -D/B, 0])
    else:
        t = np.array([-D/A, 0, 0])
    
    translated_pts = [np.array(p) - t for p in points]
    
    # Step 2: Compute rotation matrix
    n = np.array([A, B, C])
    n_norm = n / np.linalg.norm(n)
    a, b, c = n_norm
    
    denominator_ab = np.sqrt(a**2 + b**2)
    if denominator_ab < 1e-12:
        R = np.eye(3)  # No rotation needed
    else:
        cos_theta = c
        sin_theta = np.sqrt(a**2 + b**2)
        u1 = b / denominator_ab
        u2 = -a / denominator_ab
        
        R = np.array([
            [cos_theta + u1**2*(1 - cos_theta), u1*u2*(1 - cos_theta), u2*sin_theta],
            [u1*u2*(1 - cos_theta), cos_theta + u2**2*(1 - cos_theta), -u1*sin_theta],
            [-u2*sin_theta, u1*sin_theta, cos_theta]
        ])
    
    # Step 3: Rotate points
    rotated_pts = [R.dot(p) for p in translated_pts]
    
    return rotated_pts, t, R


import numpy as np

def fit_plane_to_points(points):
    """
    Adjusts plane coefficients to best fit a set of 3D points.
    """
    # Center points by subtracting centroid
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # Compute SVD of centered points matrix
    U, S, Vt = np.linalg.svd(centered)
    
    # Normal vector is last column of V (smallest singular value)
    A, B, C = Vt[-1]
    
    # Calculate D using centroid
    D = -np.dot(np.append(centroid, 1), [A, B, C, 0])
    
    # Normalize coefficients (optional)
    norm = np.linalg.norm([A, B, C])
    return A/norm, B/norm, C/norm, D/norm



def plot_planes(center_and_neighbors, supercell_points):
    plt.close()
    nrows = ceil(len(center_and_neighbors) / 3)
    # fig, ax = plt.subplots(ncols=3, nrows=nrows, figsize=(15, 48))
    
    supercell_points_c = np.vstack([p[:-1] for p in supercell_points])
    
    row = -1
    fig, ax = plt.subplots(nrows=len(center_and_neighbors), ncols=1, figsize=(15, 48))
    
    for i in range(len(center_and_neighbors)):
        plane_coeff, site, center, neighbors = center_and_neighbors[i]

        same_site_in_sc = np.vstack([p[:-1] for p in supercell_points if p[-1]==site], dtype=np.float32)
        print(type(same_site_in_sc), same_site_in_sc.dtype, same_site_in_sc.shape)
        _, same_site_in_sc = num_atoms_on_plane(plane_coeff, same_site_in_sc, 0.5)
        plane_coeff = fit_plane_to_points(same_site_in_sc)
        
        points = neighbors.tolist()
        points.append(center)
        rotated_points, T, R = align_plane_to_xy(plane_coeff, points)
        rotated_points = np.array(rotated_points)
        
        # rotate supercell
        rotated_supercellpoints = (supercell_points_c.copy() - T).dot(R)
        rotated_supercellpoints = rotated_supercellpoints[np.isclose(rotated_supercellpoints[:, 2], 0., atol=1e-1)]
        
        im = ax[i].scatter(rotated_supercellpoints[:, 0], rotated_supercellpoints[:, 1], c=rotated_supercellpoints[:, 2], cmap='viridis')
        aspect_ratio = (rotated_supercellpoints[:, 0].max() - rotated_supercellpoints[:, 0].min()) / (rotated_supercellpoints[:, 1].max() - rotated_supercellpoints[:, 1].min())
        ax[i].set_aspect(aspect_ratio)
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        
        ticks = sorted(set([round(i, 3) for i in np.linspace(rotated_supercellpoints[:, 2].min(), rotated_supercellpoints[:, 2].max(), 8)]))
        cbar = fig.colorbar(im, ticks=ticks, location='top', 
                        shrink=0.25, orientation='horizontal') # cax=ax.inset_axes((0.19, 0.72, 0.4, 0.02))
        cbar.ax.tick_params(labelsize=5) 

    #     col = i % 3
    #     if col == 0:
    #         row += 1
    #     # aspect_ratio = (rotated_points[:, 0].max() - rotated_points[:, 0].min()) / (rotated_points[:, 1].max() - rotated_points[:, 1].min())
    #     aspect_ratio = "equal"
        
    #     if nrows == 1:
    #         ax[col].set_aspect(aspect_ratio)
    #         for j in range(len(rotated_points)):
    #             ax[col].scatter(rotated_points[j, 0], rotated_points[j, 1])
    #             ax[col].text(rotated_points[j, 0]+0.1, rotated_points[j, 1], s=f"{rotated_points[j, 2]:.2f}")
    #             # ax[i].set_axis_off()
    #             ax[col].set_xticklabels([])
    #             ax[col].set_yticklabels([])
    #             ax[col].set_xticks([])
    #             ax[col].set_yticks([])
            
    #     else:
    #         ax[row, col].set_aspect(aspect_ratio)
    #         for j in range(len(rotated_points)):
    #             ax[row, col].scatter(rotated_points[j, 0], rotated_points[j, 1])
    #             ax[row, col].text(rotated_points[j, 0]+0.1, rotated_points[j, 1], s=f"{rotated_points[j, 2]:.2f}")
    #             # ax[row, col].set_axis_off()
    #             ax[row, col].set_xticklabels([])
    #             ax[row, col].set_yticklabels([])
    #             ax[row, col].set_xticks([])
    #             ax[row, col].set_yticks([])
                
    # for i in range(len(center_and_neighbors), nrows*3):
    #     col = i % 3
    #     if col == 0:
    #         row += 1
    #     if nrows == 1:
    #         ax[col].axis('off')
    #     else:
    #         ax[row, col].axis('off')
    
    # fig.subplots_adjust(wspace=0, hspace=-0.8)
    # plt.savefig('hex.png')
    return fig


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    
        cif = Cif(temp_file_path)
        unitcell_points = cif.unitcell_points
        supercell_points = cif.supercell_points
        unitcell_lengths = cif.unitcell_lengths
        unitcell_angles = cif.unitcell_angles
        
        
        for i in range(len(unitcell_points)):
            x, y, z, l = unitcell_points[i]
            cart = unit.fractional_to_cartesian([x, y, z],
                                                unitcell_lengths,
                                                unitcell_angles)
            unitcell_points[i] = (*cart, l)

        # sites
        cif.compute_connections()
        site_conns = cif.connections

        for i in range(len(supercell_points)):
            x, y, z, l = supercell_points[i]
            cart = unit.fractional_to_cartesian([x, y, z],
                                                unitcell_lengths,
                                                unitcell_angles)
            supercell_points[i] = (*cart, l)

        site_planes, planes_data = get_unique_planes(site_conns, nmax=12)
        
        
        fig = plot_planes(planes_data, supercell_points)
        st.pyplot(fig)
        