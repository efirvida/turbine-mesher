import yaml
from ..models import Hub


def read_hub_data(yaml_file):
    with open(yaml_file) as blade_yaml:
        # data = yaml.load(blade_yaml,Loader=yaml.FullLoader)
        data = yaml.load(blade_yaml, Loader=yaml.Loader)

    # obtain hub outer shape bem
    try:
        return Hub(data["components"]["hub"]["outer_shape_bem"])
    except KeyError:
        # older versions of wind ontology do not have 'outer_shape_bem' subsection for hub data
        return Hub(data["components"]["hub"])
