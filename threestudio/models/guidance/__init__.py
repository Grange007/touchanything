from importlib import import_module

from . import stable_diffusion_guidance


_OPTIONAL_GUIDANCE_MODULES = (
    "deep_floyd_guidance",
    "stable_diffusion_dreamtimes_guidance",
    "stable_diffusion_vsd_guidance",
    "threefuse_vsd_guidance",
    "threefuse_sd_guidance",
)


for _module_name in _OPTIONAL_GUIDANCE_MODULES:
    try:
        import_module(f"{__name__}.{_module_name}")
    except ModuleNotFoundError:
        # Some guidance backends require optional model packages. The default
        # TouchAnything demo only needs stable_diffusion_guidance.
        pass
