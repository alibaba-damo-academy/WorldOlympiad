"""Declarative rigid apparatus for physical reference prototypes.

These are procedural fixtures, not new SceneCode-generated assets.
"""
from pathlib import Path

from .models import BodyIR, Geometry, Inertial, LinkIR, Pose


EXPERIMENTS = ('ramp_roll', 'domino_chain', 'collision_chain', 'shallow_containment')


def solid(name, *, color, size=None, radius=None, static=False, mass=.15):
    if radius is not None:
        geometry=Geometry(name='shape',kind='sphere',radius=radius,rgba=color)
        inertia=(.4*mass*radius*radius,)*3
    else:
        geometry=Geometry(name='shape',kind='box',size=size,rgba=color)
        x,y,z=size
        inertia=(mass*(y*y+z*z)/12,mass*(x*x+z*z)/12,mass*(x*x+y*y)/12)
    return BodyIR(name=name,semantic_category='physical_apparatus',static=static,
        links=[LinkIR(name='body',inertial=Inertial(mass=mass,inertia=(*inertia,0,0,0)),visuals=[geometry],collisions=[geometry.model_copy(deep=True)])],
        metadata={'asset_source':'procedural_physical_fixture','visual_mesh_coordinate_frame':'worldir_z_up','simready_visual_rgba':list(color)})


def apparatus(task_id, output):
    """Return bodies, bottom-center placements relative to table, and triggers."""
    teal=(.03,.5,.48,1);red=(.8,.08,.045,1);blue=(.06,.24,.7,1);yellow=(.95,.65,.03,1)
    bodies=[];layout={};trigger={};subject='subject';grounding=''
    if task_id=='ramp_roll':
        import trimesh
        path=Path(output)/'apparatus/ramp.obj';path.parent.mkdir(parents=True,exist_ok=True)
        # Highest edge at -X; the continuous top slopes down toward +X.
        vertices=[[-.25,-.09,0],[-.25,.09,0],[.25,-.09,0],[.25,.09,0],[-.25,-.09,.15],[-.25,.09,.15]]
        faces=[[0,2,3],[0,3,1],[0,1,5],[0,5,4],[0,4,2],[1,3,5],[4,5,3],[4,3,2]]
        trimesh.Trimesh(vertices=vertices,faces=faces,process=True).export(path)
        ramp=solid('ramp',color=(.5,.28,.11,1),size=(.5,.18,.15),static=True)
        g=Geometry(name='wedge',kind='mesh',path=str(path),rgba=(.5,.28,.11,1))
        ramp.links[0].visuals=[g];ramp.links[0].collisions=[g.model_copy(deep=True)]
        # A physical run-out stop keeps the complete experiment in view.
        stop=solid('runout_stop',color=blue,size=(.025,.22,.07),static=True)
        bodies=[solid(subject,color=teal,radius=.04),ramp,stop]
        layout={'ramp':[-.15,0,0],subject:[-.35,0,.144],'runout_stop':[.50,0,0]}
        grounding='teal ball';trigger={'kind':'gravity_release'}
    elif task_id=='domino_chain':
        for index in range(6):
            name=subject if index==0 else f'domino_{index+1}'
            bodies.append(solid(name,color=red if index%2==0 else blue,size=(.035,.07,.16),mass=.08))
            layout[name]=[-.3+index*.095,0,0]
        grounding='first red upright block'
        trigger={'kind':'push','body':subject,'start_s':.25,'end_s':.32,'force_per_mass':4.,'height_fraction':.95}
    elif task_id=='collision_chain':
        for name,color,x in [(subject,red,-.34),('ball_blue',blue,-.01),('ball_yellow',yellow,.09)]:
            bodies.append(solid(name,color=color,radius=.045));layout[name]=[x,0,0]
        grounding='red ball';trigger={'kind':'velocity','body':subject,'velocity':[.65,0,0]}
    elif task_id=='shallow_containment':
        purple=(.47,.1,.7,1);cream=(.8,.72,.55,1)
        bodies.append(solid(subject,color=purple,radius=.04));layout[subject]=[.03,0,.28]
        for name,size,pos in [
            ('tray_bottom',(.30,.24,.012),(0,0,0)),
            ('tray_left',(.012,.24,.04),(-.144,0,.012)),
            ('tray_right',(.012,.24,.04),(.144,0,.012)),
            ('tray_back',(.276,.012,.04),(0,.114,.012)),
            ('tray_front',(.276,.012,.04),(0,-.114,.012)),
        ]:
            bodies.append(solid(name,color=cream,size=size,static=True));layout[name]=list(pos)
        grounding='purple ball';trigger={'kind':'gravity_release'}
    else:
        raise ValueError(f'unsupported physical apparatus: {task_id}')
    return bodies,layout,trigger,grounding
