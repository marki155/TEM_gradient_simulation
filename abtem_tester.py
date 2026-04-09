import numpy as np
import torch
import ase
import ase.build
import abtem


def build_same_atoms_as_main(H, W):

    fov = 7.58
    pixel_size_nm = fov / W



    single_WS2 = ase.build.mx2(formula='WS2')
    WS2_H_atoms = int((H * pixel_size_nm * 10) / single_WS2.cell.lengths()[0]) * 2
    WS2_W_atoms = int((W * pixel_size_nm * 10) / single_WS2.cell.lengths()[1]) * 2

    single_graphen = ase.build.graphene()
    graphen_H_atoms = int((H * pixel_size_nm * 10) / single_graphen.cell.lengths()[0]) * 2
    graphen_W_atoms = int((W * pixel_size_nm * 10) / single_graphen.cell.lengths()[1]) * 2

    ws2 = ase.build.mx2(formula='WS2', vacuum=1.0) * (WS2_H_atoms, WS2_W_atoms, 1)
    ws2.translate((20, -90, 0))
    ws2.rotate(60, 'z')

    graphen = ase.build.graphene(vacuum=1.0) * (graphen_H_atoms, graphen_W_atoms, 1)
    graphen.rotate(0, 'z')
    graphen.translate((-10, -18, 0))

    return graphen, ws2


def crop_atoms_to_canvas(atoms, H, W, pixel_size_nm):
    pos_px = atoms.get_positions() / (pixel_size_nm * 10.0)

    selector = (
        (pos_px[:, 0] >= 0) &
        (pos_px[:, 0] <= H) &
        (pos_px[:, 1] >= 0) &
        (pos_px[:, 1] <= W)
    )

    cropped = atoms[selector]
    return cropped


def combine_graphene_ws2_for_abtem(H, W, pixel_size_nm,
                                   graphene_z_nm=0.0,
                                   ws2_bottom_s_z_nm=0.33):
    graphen, ws2 = build_same_atoms_as_main(H, W)

    graphen = crop_atoms_to_canvas(graphen, H, W, pixel_size_nm)
    ws2 = crop_atoms_to_canvas(ws2, H, W, pixel_size_nm)

    # absolute z sauber setzen:
    # Graphen auf z = graphene_z_nm
    gpos = graphen.get_positions().copy()
    gpos[:, 2] = graphene_z_nm * 10.0
    graphen.set_positions(gpos)

    # WS2 relativ zu seiner unteren S-Ebene auf ws2_bottom_s_z_nm
    wpos = ws2.get_positions().copy()
    zmin = wpos[:, 2].min()
    zmax = wpos[:, 2].max()
    wpos[:, 2] = (wpos[:, 2] - zmin)# + ws2_bottom_s_z_nm * 10.0
    ws2.set_positions(wpos)

    atoms = ws2

    # gemeinsame Zellgröße passend zum simulierten Bild
    Lx = W * pixel_size_nm * 10.0
    Ly = H * pixel_size_nm * 10.0
    zmax = atoms.get_positions()[:, 2].max()
    Lz = zmax#max(zmax + 20.0, 40.0)  # Å, genug Vakuum

    atoms.set_cell([Lx, Ly, Lz])
    atoms.set_pbc([False, False, False])

    return atoms


def simulate_abtem_lobato_target(
    atoms,
    H,
    W,
    pixel_size_nm,
    voltage_kV,
    defocus_nm=-57.0,
    cs_mm=2.0,
    focal_spread_nm=0.0,
    angular_spread_mrad=0.0,
    slice_thickness_nm=0.2,
):
    """
    Liefert ein HxW Float-Bild als NumPy-Array.
    """
    sampling_A = pixel_size_nm * 10.0
    slice_thickness_A = slice_thickness_nm * 10.0

    potential = abtem.Potential(
        atoms,
        gpts=(H, W),
        sampling=sampling_A,
        parametrization="lobato",
        slice_thickness=slice_thickness_A,
        projection="finite",
    )

    wave = abtem.PlaneWave(
        energy=voltage_kV * 1e3
    )

    exit_wave = wave.multislice(potential)

    abb_dict = {"C10": -13.07*10,
                               'C12': 0, "phi12": 0.,
                               'C23': 0, 'phi23': 0.,
                               'C21': 0, 'phi21': 0.,
                               "C30": 4.16*1e4, #um
                               'C34': 0, 'phi34': 0., #um
                               'C45': 0, 'phi45': 0.,
                                'Cs': 1.e3 * 1e4}  # um -> Å

    ctf = abtem.CTF(
        energy=voltage_kV * 1e3,
        aberration_coefficients=abb_dict,  # mm -> Å
        focal_spread=focal_spread_nm * 100.0,   # nm -> Å
        angular_spread=angular_spread_mrad,
    )

    image = exit_wave.apply_ctf(ctf).intensity().compute().array

    if image.shape != (H, W):
        image = np.asarray(image).reshape(H, W)

    image = image.astype(np.float32)
    image -= image.min()
    if image.max() > 0:
        image /= image.max()

    return image


def make_abtem_target_from_main_geometry(
    H,
    W,
    pixel_size_nm,
    voltage_kV,
    defocus_nm=-13.0,
    cs_mm=2.0,
    focal_spread_nm=0.0,
    angular_spread_mrad=0.0,
    graphene_z_nm=0.0,
    ws2_bottom_s_z_nm=0.33,
    slice_thickness_nm=0.13,
):
    atoms = combine_graphene_ws2_for_abtem(H, W, pixel_size_nm)


    image = simulate_abtem_lobato_target(
        atoms=atoms,
        H=H,
        W=W,
        pixel_size_nm=pixel_size_nm,
        voltage_kV=voltage_kV,
        defocus_nm=defocus_nm,
        cs_mm=cs_mm,
        focal_spread_nm=focal_spread_nm,
        angular_spread_mrad=angular_spread_mrad,
        slice_thickness_nm=slice_thickness_nm,
    )

    return image, atoms



import tifffile
import torch
import numpy as np
import matplotlib.pyplot as plt

from abtem_tester import make_abtem_target_from_main_geometry

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEVICE)

VOLTAGE_KV = 80

if __name__ == '__main__':
    # nur um H/W/FOV zu bekommen
    image_ref = tifffile.imread('NPL11-1_crop_big_blur-3.tif')
    image_ref[np.where(image_ref == 0.0)] = np.median(image_ref)

    H, W = image_ref.shape
    fov = 7.58
    pixel_size_nm = fov / W

    simulated_target_np, abtem_atoms = make_abtem_target_from_main_geometry(
        H=H,
        W=W,
        pixel_size_nm=pixel_size_nm,
        voltage_kV=VOLTAGE_KV,
        defocus_nm=-100.0,
        cs_mm=2.0,
        focal_spread_nm=0.35,
        angular_spread_mrad=0.000005,
        graphene_z_nm=0.0,
        ws2_bottom_s_z_nm=0.33,
        slice_thickness_nm=0.2,
    )

    # optional speichern
    tifffile.imwrite("abtem_target_lobato.tif", simulated_target_np)

    target = torch.tensor(simulated_target_np, dtype=torch.float32)

    brightness = torch.mean(target)
    contrast = torch.std(target)

    # ab hier dein restlicher Code praktisch unverändert
    plt.imshow(simulated_target_np)
    plt.show()

