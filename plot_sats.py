import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from sharc.satellite.ngso.orbit_model import OrbitModel
from sharc.ngso_to_gso import _build_gso_link_context, _group_contexts_by_earth_station
from sharc.satellite.scripts.plot_globe import plot_globe_with_borders
from sharc.support.sharc_geom import CoordinateSystem
from sharc.support.geometry import (
    SimulatorGeometry, ENUReferenceFrame, DWNReferenceFrame, plot_geom
)
from sharc.parameters.parameters_ngso_to_gso import (
    ParametersNGSO2GSO, ParametersGSO, STR_SEPARATOR
)
from sharc.support.sharc_geom import polar_to_cartesian

ALPHA = 3.0

# def plot_scenario(
#     global_coord_sys: CoordinateSystem,
#     gso_es_geom: SimulatorGeometry,
#     gso_ss_geom: SimulatorGeometry,
#     ngso_geom: SimulatorGeometry,
# ):
#     return fig

def make_cone(
    origin,
    direction,
    length,
    angle_deg=3.0,
    n_segments=32,
    color="red",
    opacity=0.2,
    show_base=False,
):
    """
    Create a Plotly Mesh3d cone.

    Parameters
    ----------
    origin : (3,) array_like
        Cone apex.
    direction : (3,) array_like
        Cone axis (doesn't need to be normalized).
    length : float
        Cone length.
    angle_deg : float
        Cone half-angle in degrees.
    n_segments : int
        Number of vertices around the base.
    show_base : bool
        Whether to close the base.

    Returns
    -------
    go.Mesh3d
    """

    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)

    direction /= np.linalg.norm(direction)

    # Find a vector not parallel to direction
    if abs(direction[2]) < 0.9:
        tmp = np.array([0., 0., 1.])
    else:
        tmp = np.array([0., 1., 0.])

    u = np.cross(direction, tmp)
    u /= np.linalg.norm(u)

    v = np.cross(direction, u)

    radius = length * np.tan(np.deg2rad(angle_deg))

    center = origin + length * direction

    theta = np.linspace(0, 2*np.pi, n_segments, endpoint=False)

    ring = (
        center
        + radius * np.cos(theta)[:, None] * u
        + radius * np.sin(theta)[:, None] * v
    )

    vertices = np.vstack([origin, ring])

    if show_base:
        vertices = np.vstack([vertices, center])
        center_idx = len(vertices) - 1

    i = []
    j = []
    k = []

    # Side triangles
    for t in range(n_segments):
        a = 0
        b = t + 1
        c = ((t + 1) % n_segments) + 1

        i.append(a)
        j.append(b)
        k.append(c)

    # Base triangles
    if show_base:
        for t in range(n_segments):
            b = t + 1
            c = ((t + 1) % n_segments) + 1

            i.append(center_idx)
            j.append(c)
            k.append(b)

    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=opacity,
        flatshading=True,
    )

def main():
    param_file = Path("./scenario2.yaml")
    par = ParametersNGSO2GSO()
    par.load_parameters_from_file(param_file)
    par.validate("ngso2gso")

    orbit_models = [
        OrbitModel(
            Nsp=p.sats_per_plane, Np=p.n_planes,
            phasing=p.phasing_deg, long_asc=p.long_asc_deg,
            omega=p.omega_deg, delta=p.inclination_deg,
            hp=p.perigee_alt_km, ha=p.apogee_alt_km,
            Mo=p.initial_mean_anomaly,
            # IGNORE THIS
            model_time_as_random_variable=False,
            t_min=0, t_max=-1,
        )
        for p in par.ngso.orbits
    ]
    total_sats = sum(p.n_planes * p.sats_per_plane for p in par.ngso.orbits)

    # global_coord_sys = CoordinateSystem()
    # global_coord_sys.set_reference(0.0, -100., 0.)

    gso_contexts = [
        _build_gso_link_context(gso_par)
        for gso_par in par.gso_links
    ]
    es_groups = _group_contexts_by_earth_station(gso_contexts)

    timeline = np.linspace(0, 500, 500)
    n_steps = len(timeline)
    ctx = gso_contexts[0]
    global_coord_sys = ctx.global_coord_sys
    # --- Batch orbit propagation for this chunk ---
    all_orbit_positions = [
        # testing random positions
        # o.get_orbit_positions_random(orbital_rng, len(chunk_timeline))
        o.get_orbit_positions_time_instant(timeline)
        for o in orbit_models
    ]
    # (len(timeline), total_sats)
    ngso_x_ecef = np.vstack([pos['sx'] for pos in all_orbit_positions]) * 1e3
    ngso_y_ecef = np.vstack([pos['sy'] for pos in all_orbit_positions]) * 1e3
    ngso_z_ecef = np.vstack([pos['sz'] for pos in all_orbit_positions]) * 1e3

    ############
    # FIXED plot information
    fig = plot_globe_with_borders(True, global_coord_sys, False)
    for ctx in gso_contexts[:1]:
        pointng_at = np.ravel(
            polar_to_cartesian(1, ctx.es_geom.pointn_azim_global, ctx.es_geom.pointn_elev_global)
        )
        fig.add_trace(
            make_cone(
                np.ravel((ctx.es_geom.x_global, ctx.es_geom.y_global, ctx.es_geom.z_global)),
                pointng_at, 1e6, ALPHA
            )
        )
        plot_geom(
            fig, ctx.es_geom, plot_pointing=True,
            scatter_params={
                "marker": dict(size=3, color='green', opacity=1),
            }, boresight_length=400 * 1e3
        )
        plot_geom(fig, ctx.ss_geom)

    ############
    # per frame shit
    # initial positions
    nx, ny, nz = global_coord_sys.ecef2enu(
        ngso_x_ecef[:, 0], ngso_y_ecef[:, 0], ngso_z_ecef[:, 0]
    )
    ngso_all_trace = go.Scatter3d(
        x=nx, y=ny, z=nz,
        mode="markers",
        name="Non selectable NGSO (off_axis < arc_avoidance AND elevation<35deg)",
        marker=dict(
            size=1,
            color="red",
        ),
    )
    ngso_geom = SimulatorGeometry(total_sats)
    ngso_geom.set_global_coords(nx, ny, nz)
    off_axis = ctx.es_geom.get_off_axis_angle(ngso_geom)[0]
    elevation = ctx.es_geom.get_local_elevation(ngso_geom)[0]
    msk = (off_axis > ALPHA) & (elevation > 35.)
    ngso_active_trace = go.Scatter3d(
        x=nx[msk], y=ny[msk], z=nz[msk],
        mode="markers",
        name="Selectable NGSO",
        marker=dict(
            size=2,
            color="yellow",
        ),
    )

    fig.add_trace(ngso_all_trace)
    fig.add_trace(ngso_active_trace)

    ############
    # Animation frames

    frames = []

    trace_active_idx = len(fig.data) - 1
    trace_all_idx = len(fig.data) - 2

    for i in range(n_steps):
        nx, ny, nz = global_coord_sys.ecef2enu(
            ngso_x_ecef[:, i],
            ngso_y_ecef[:, i],
            ngso_z_ecef[:, i],
        )
        ngso_geom = SimulatorGeometry(total_sats)
        ngso_geom.set_global_coords(nx, ny, nz)
        off_axis = ctx.es_geom.get_off_axis_angle(ngso_geom)[0]
        elevation = ctx.es_geom.get_local_elevation(ngso_geom)[0]
        msk = (off_axis > ALPHA) & (elevation > 35.)

        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=nx,
                        y=ny,
                        z=nz,
                    ),
                    go.Scatter3d(
                        x=nx[msk], y=ny[msk], z=nz[msk],
                    )
                ],
                traces=[trace_all_idx, trace_active_idx],
                name=str(i),
            )
        )

    fig.frames = frames

    ############
    # plot config
    # Set the camera position in Plotly
    # show_range = 1e4
    show_range = 8e7
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            aspectmode="cube",
            zaxis=dict(
                range=(-show_range / 2, show_range / 2)
            ),
            yaxis=dict(
                range=(-show_range / 2, show_range / 2)
            ),
            xaxis=dict(
                range=(-show_range / 2, show_range / 2)
            ),
            camera=dict(
                # center=dict(x=0, y=0, z=center_of_earth.geom.z_global[0] / show_range / 1e3),
                # eye=dict(x=0, y=0, z=0.7),  # Eye position (above the center)
                # up=dict(x=0, y=1, z=0)      # "Up" is along +y (default is usually +z)
            )
        ),
        legend=dict(
            x=0.02,        # Move to left
            y=0.02,        # Near top
            bgcolor='rgba(255,255,255,1)',  # Optional: semi-transparent background
            bordercolor='black',
            borderwidth=1
        ),
        updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=10, redraw=True),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                    mode="immediate",
                                ),
                            ],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0),
                                    transition=dict(duration=0),
                                    mode="immediate",
                                ),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    currentvalue=dict(prefix="Frame: "),
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [str(i)],
                                dict(
                                    mode="immediate",
                                    frame=dict(duration=0, redraw=False),
                                    transition=dict(duration=0),
                                ),
                            ],
                            label=str(i),
                        )
                        for i in range(n_steps)
                    ],
                )
            ],
        # width=700,
        # height=700,
    )
    fig.show()


if __name__ == "__main__":
    main()
