"""3-D animations, one module per topic. Import from here:
    from visualization.animate import animate_d2min_3d
"""

from visualization.animate.holes import animate_holes_KDTree, animate_holes_fft
from visualization.animate.defects import animate_defects
from visualization.animate.d2min import (
    animate_d2min_2d,
    animate_d2min_2d_cumulative,
    animate_d2min_3d,
    animate_d2min_3d_cumulative,
    animate_d2min_defects_overlay_3d,
    animate_d2min_holes_overlay_3d,
)
from visualization.animate.softness import (
    animate_softness_3d,
    animate_distribution_by_element,
    animate_d2min_softness_side_by_side,
)
