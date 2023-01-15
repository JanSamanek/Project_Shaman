import fiftyone.zoo as foz
import fiftyone as fo
from fiftyone import ViewField as F

labels_path = r"C:\Users\jands\fiftyone\coco-2017\train\labels"
label_field = "ground_truth"

dataset = foz.load_zoo_dataset(
    "coco-2017",
    splits=('train', 'validation', 'test'),
    label_types=["detections"],
    shuffle=True,
    classes=["person"],
    max_samples=3500,
)

# filter labels present from coco dataset
dataset = dataset.filter_labels("ground_truth", F("label").is_in(["person"]))

split_view = dataset.match_tags('train')

split_view.export(
    dataset_type=fo.types.YOLOv5Dataset,
    labels_path=labels_path,
    label_field=label_field,
    split='train'
)

labels_path = r"C:\Users\jands\fiftyone\coco-2017\test\labels"
split_view = dataset.match_tags('test')

split_view.export(
    dataset_type=fo.types.YOLOv5Dataset,
    labels_path=labels_path,
    label_field=label_field,
    split='test'
)

labels_path = r"C:\Users\jands\fiftyone\coco-2017\validation\labels"
split_view = dataset.match_tags('validation')

split_view.export(
    dataset_type=fo.types.YOLOv5Dataset,
    labels_path=labels_path,
    label_field=label_field,
    split='validation'
)