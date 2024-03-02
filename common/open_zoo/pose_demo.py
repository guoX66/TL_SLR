from time import perf_counter
from common.open_zoo.models import ImageModel
from common.open_zoo.pipelines import get_user_config, AsyncPipeline
from common.open_zoo.adapters import create_core, OpenvinoAdapter


def ov_ini(model, frame, device):
    ie = create_core()
    plugin_config = get_user_config(device, '', 6)
    model_adapter = OpenvinoAdapter(ie, model, device=device, plugin_config=plugin_config,
                                    max_num_requests=0,
                                    model_parameters={'input_layouts': None})

    config = {
        'target_size': None,
        'aspect_ratio': frame.shape[1] / frame.shape[0],
        'confidence_threshold': 0.1,
        'padding_mode': None,
        # the 'higherhrnet' and 'ae' specific
        'delta': None,  # the 'higherhrnet' and 'ae' specific
    }
    model = ImageModel.create_model('HPE-assosiative-embedding', model_adapter, config)
    # model.log_layers_info()
    hpe_pipeline = AsyncPipeline(model)
    return hpe_pipeline, ie


def pose_inference(hpe_pipeline, frame):
    start_time = perf_counter()
    hpe_pipeline.submit_data(frame, 0, {'frame': frame, 'start_time': start_time})
    # Process all completed requests
    results = hpe_pipeline.get_result(0)
    return results
