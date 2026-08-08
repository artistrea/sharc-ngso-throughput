from sharc.antenna.antenna_beamforming_imt import AntennaBeamformingImt
from sharc.parameters.imt.parameters_antenna_imt import ParametersAntennaImt
from sharc.parameters.parameters import Parameters

from pathlib import Path


SHARC_ROOT_DIR = Path(__file__).parent

if __name__ == "__main__":
    ######################################################
    # config:
    USE_FILE = False
    param_file = (
        SHARC_ROOT_DIR
            # / ".."
        # / "input"
        / "parameters_mss_d2d_to_imt_ul_co_channel_AAS_k_1_system_A.yaml"
    )

    PLOT_SPHERICAL = False
    PLOT_PLAIN = True

    default_surface_config = dict(
        colorscale='Turbo',
        # colorscale='Viridis',
        # colorscale='Jet',
        colorbar=dict(title='Gain [dB]'),
        # colorbar=None,
        showscale=False,
        # colorbar range
        cmin=-60,
        cmax=30,
        # to remove light from changing the truecolors:
        lighting=dict(
            ambient=1.0,
            diffuse=0.0,
            specular=0.0,
            roughness=1.0,
            fresnel=0.0
        ),
        lightposition=dict(
            x=0,
            y=0,
            z=0
        ),
    )

    if not USE_FILE:
        aspectmode = 'manual',
        aspectratio = dict(x=1, y=1, z=0.5),  # z adjusted to your liking
        bs_param = ParametersAntennaImt()
        bs_param.normalization = False
        # bs_param.normalization_file = 'beamforming_normalization\\bs_indoor_norm.npz'
        bs_param.minimum_array_gain = -60

        bs_param.element_pattern = "M2101"
        bs_param.element_max_g = 6.5
        bs_param.element_phi_3db = 65
        bs_param.element_theta_3db = 90
        bs_param.element_am = 30
        bs_param.element_sla_v = 30
        bs_param.n_rows = 8
        bs_param.n_columns = 8
        bs_param.element_horiz_spacing = 0.5
        bs_param.element_vert_spacing = 0.5
        bs_param.multiplication_factor = 12

        bs_param.downtilt = 10

        # bs_param.subarray.is_enabled = True
        # bs_param.subarray.element_vert_spacing = ?
        # bs_param.subarray.eletrical_downtilt = 3
        # bs_param.subarray.n_rows = 3
    else:
        param_file = param_file.resolve()
        print("File at:")
        print(f"  '{param_file}'")

        parameters = Parameters()
        parameters.set_file_name(param_file)
        parameters.read_params()

        bs_param = parameters.imt.bs.antenna.array

    antenna = AntennaBeamformingImt(
        bs_param.get_antenna_parameters(),
        0,
        -bs_param.downtilt,
    )

    # beam/electrical tilt in spherical coords
    # phi is azim
    phi_escan = 0
    # theta is elev
    theta_tilt = 90
    antenna.add_beam(phi_escan, theta_tilt)

    import numpy as np
    import plotly.graph_objects as go

    # Step 1: Create phi and theta grids
    phi = np.linspace(-180, 180, 360)  # azimuth in degrees
    theta = np.linspace(-90, 90, 180)   # elevation in degrees
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    # Step 2: Convert to radians for math
    phi_rad = np.radians(phi_grid)
    theta_rad = np.radians(theta_grid)

    # Step 3: Simulate or call antenna gain function
    # Replace this with your actual function
    # def dummy_gain(phi_deg, theta_deg):
    # return np.abs(np.cos(np.radians(theta_deg))) *
    # np.abs(np.cos(np.radians(phi_deg)))

    # gain = dummy_gain(phi_grid, theta_grid)
    gain = antenna.calculate_gain(
        phi_vec=np.ravel(phi_grid),
        theta_vec=90 - np.ravel(theta_grid),
        beams_l=np.zeros_like(np.ravel(phi_grid), dtype=int),
    )
    gain = np.reshape(gain, phi_grid.shape)

    if PLOT_SPHERICAL:
        # Step 4: Convert spherical (r=gain, theta, phi) to Cartesian (x, y, z)
        r = np.power(10, 0.1 * gain)
        x = r * np.sin(-(theta_rad - np.pi / 2)) * np.cos(phi_rad)
        y = r * np.sin(-(theta_rad - np.pi / 2)) * np.sin(phi_rad)
        z = r * np.cos(-(theta_rad - np.pi / 2))

        # Step 5: Create 3D surface plot
        fig = go.Figure(data=[go.Surface(
            x=x, y=y, z=z,
            surfacecolor=gain,
            **default_surface_config,
        )])
        fig.update_layout(
            title=f"Ant Gain; elev_etilt={theta_tilt}; azim_etilt={phi_escan};",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data"),
            margin=dict(
                l=0,
                r=0,
                t=30,
                b=0))
        fig.show()

    if PLOT_PLAIN:
        heat_surface = go.Surface(
            x=phi_grid,
            y=theta_grid,
            z=gain,
            # z=np.power(10, 0.1*gain),
            surfacecolor=gain,          # color mapped to gain
            **default_surface_config,
        )
        # Create wireframe lines manually
        grid_lines = []

        # Horizontal (theta constant)
        for i in range(0, theta_grid.shape[0], 2):
            grid_lines.append(go.Scatter3d(
                x=phi_grid[i, :],
                y=theta_grid[i, :],
                z=gain[i, :],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False
            ))

        # Vertical (phi constant)
        for j in range(0, phi_grid.shape[1], 2):
            grid_lines.append(go.Scatter3d(
                x=phi_grid[:, j],
                y=theta_grid[:, j],
                z=gain[:, j],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False
            ))

        # Combine surface and wireframe
        fig = go.Figure(data=[heat_surface] + grid_lines)

        fig.update_layout(
            title='',
            scene=dict(
                xaxis_title='Azimuth φ [deg]',
                yaxis_title='Elevation θ [deg]',
                zaxis_title='Gain',
                xaxis=dict(dtick=50, range=[-200, 200]),
                yaxis=dict(dtick=20, range=[-100, 100]),
                # zaxis=dict(nticks=5, range=[-60, 30]),
                zaxis=dict(
                           # showticklabels=False,
                           dtick=10,
                           title='',
                           range=[-60, 30]),  # clean look
                aspectratio=dict(x=1,y=4/5,z=4/5),
                camera=dict(
                    # projection=dict(type="perspective"),
                    eye=dict(x=-1.2, y=-1.2, z=0.5),
                )
            ),
            # margin=dict(l=0, r=0, t=50, b=0),
            # width=500,
            # height=500
        )
        fig.show()

# th = (0,180)
# nth = 90 - th # (90, -90)
