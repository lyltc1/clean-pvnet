from .resnet18  import get_res_epnet


_network_factory = {
    'res': get_res_epnet
}


def get_network(cfg):
    arch = cfg.network
    get_model = _network_factory[arch]
    network = get_model(cfg.heads['num_keypoints'])
    return network
