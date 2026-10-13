def test_can_import_public_api():
    import csi.models as M  # noqa: F401
    from csi.models import core, utils, tailored, metrics  # noqa: F401
    from csi.models.utils import model_utils, gpu_utils  # noqa: F401
    from csi.models.metrics import results_tables  # noqa: F401

