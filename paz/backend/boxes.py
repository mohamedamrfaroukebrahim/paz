import cv2
import jax.numpy as jp
from jax import lax
import paz


def split_into_components(boxes):
    """Split boxes into x_min, y_min, x_max, y_max components."""
    return jp.split(boxes, 4, axis=1)


def get_center_coordinates(x_minimum, x_maximum, y_minimum, y_maximum):
    """Compute center coordinates of boxes."""
    center_x = (x_maximum + x_minimum) / 2.0
    center_y = (y_maximum + y_minimum) / 2.0
    return center_x, center_y


def get_width_and_height(x_minimum, x_maximum, y_minimum, y_maximum):
    """Compute width and height from box corners."""
    w = x_maximum - x_minimum
    h = y_maximum - y_minimum
    return w, h


def to_center_form(boxes):
    """Convert bounding boxes from corner to center form.

    Args:
        boxes (array): Boxes in corner format [x_min, y_min, x_max, y_max]

    Returns:
        array: Boxes in center format [center_x, center_y, width, height]
    """
    x_minimum, y_minimum, x_maximum, y_maximum = split_into_components(boxes)
    center_x, center_y = get_center_coordinates(x_minimum, x_maximum, y_minimum, y_maximum)
    W, H = get_width_and_height(x_minimum, x_maximum, y_minimum, y_maximum)
    return jp.concatenate([center_x, center_y, W, H], axis=1)


def to_corner_form(boxes):
    """Convert bounding boxes from center to corner form.

    Args:
        boxes (array): Boxes in center format [center_x, center_y, width, height]

    Returns:
        array: Boxes in corner format [x_min, y_min, x_max, y_max]
    """
    center_x, center_y, W, H = split_into_components(boxes)
    x_minimum = center_x - (W / 2.0)
    x_maximum = center_x + (W / 2.0)
    y_minimum = center_y - (H / 2.0)
    y_maximum = center_y + (H / 2.0)
    return jp.concatenate([x_minimum, y_minimum, x_maximum, y_maximum], axis=1)


def pad(boxes, size, value=-1):
    """Pad boxes array to specified size with value."""
    num_boxes = len(boxes)
    if num_boxes > size:
        raise ValueError(f"Samples ({num_boxes}) exceeds pad ({size}).")
    padding = ((0, size - num_boxes), (0, 0))
    return jp.pad(boxes, padding, "constant", constant_values=value)


def order_xyxy(point_A, point_B):
    """Order coordinates to form a valid bounding box."""
    x1, y1 = point_A
    x2, y2 = point_B
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def take_last_two_points(points):
    """Retrieve the last two points from a list."""
    return points[-1], points[-2]


def draw_circle_and_store_point(image, points, x, y, radius, color):
    """Draw a circle on image and store the point."""
    paz.draw.circle(image, (x, y), radius, color)
    points.append((x, y))


def process_pair_of_points(image, points, boxes, color, radius):
    """Process two points to form a box and draw it."""
    point_A, point_B = take_last_two_points(points)
    box = order_xyxy(point_A, point_B)
    paz.draw.box(image, box, color, radius)
    boxes.append(box)


def on_double_click_event(event, x, y, flags, param, image, points, boxes, radius, color):
    """Handle double-click event to draw points and boxes."""
    if event == cv2.EVENT_LBUTTONDBLCLK:
        draw_circle_and_store_point(image, points, x, y, radius, color)
        if len(points) % 2 == 0:
            process_pair_of_points(image, points, boxes, color, radius)


def create_on_double_click(image, points, boxes, radius, color):
    """Create a mouse callback function for double clicks."""

    def on_double_click(event, x, y, flags, param):
        on_double_click_event(event, x, y, flags, param, image, points, boxes, radius, color)

    return on_double_click


def setup_window_and_callback(image, points, boxes, radius, color, window_name):
    """Set up the window and mouse callback for box selection."""
    on_double_click = create_on_double_click(image, points, boxes, radius, color)
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_double_click)


def key_is_pressed(key="q", time=20):
    """Check if a specific key is pressed."""
    return cv2.waitKey(time) & 0xFF == ord(key)


def run_event_loop(image, window_name):
    """Run the main event loop for displaying the image and handling key presses.

    Args:
        image (array): The image to display
        window_name (str): Name of the display window

    Returns:
        None
    """
    while True:
        paz.image.show(image, window_name, False)
        if key_is_pressed("q"):
            break
    cv2.destroyWindow(window_name)


def convert_to_center_form(matched_boxes):
    """Convert matched boxes to center form."""
    return to_center_form(matched_boxes[:, :4])


def encode_center_coordinates(boxes, priors, variances):
    """Encode center coordinates using priors and variances."""
    difference_x = boxes[:, 0:1] - priors[:, 0:1]
    difference_y = boxes[:, 1:2] - priors[:, 1:2]
    encoded_center_x = (difference_x / priors[:, 2:3]) / variances[0]
    encoded_center_y = (difference_y / priors[:, 3:4]) / variances[1]
    return encoded_center_x, encoded_center_y


def encode_dimensions(boxes, priors, variances):
    """Encode width and height dimensions."""
    ratio_width = boxes[:, 2:3] / priors[:, 2:3]
    ratio_height = boxes[:, 3:4] / priors[:, 3:4]
    encoded_width = jp.log(ratio_width + 1e-8) / variances[2]
    encoded_height = jp.log(ratio_height + 1e-8) / variances[3]
    return encoded_width, encoded_height


def concatenate_encoded_boxes(encoded_x, encoded_y, encoded_w, encoded_h, extras):
    """Concatenate encoded box parameters with extras."""
    return jp.concatenate([encoded_x, encoded_y, encoded_w, encoded_h, extras], axis=1)


def encode(matched, priors, variances=[0.1, 0.1, 0.2, 0.2]):
    """Encode matched bounding boxes relative to prior boxes.

    Args:
        matched (array): Matched target boxes
        priors (array): Prior boxes
        variances (list): Variances for encoding (cx, cy, w, h)

    Returns:
        array: Encoded box parameters
    """
    boxes_center = convert_to_center_form(matched)
    extras = matched[:, 4:]
    encoded_x, encoded_y = encode_center_coordinates(boxes_center, priors, variances)
    encoded_w, encoded_h = encode_dimensions(boxes_center, priors, variances)
    return concatenate_encoded_boxes(encoded_x, encoded_y, encoded_w, encoded_h, extras)


def decode_center_x(predictions, priors, variances):
    """Decode center x-coordinate from predictions."""
    return predictions[:, 0:1] * priors[:, 2:3] * variances[0] + priors[:, 0:1]


def decode_center_y(predictions, priors, variances):
    """Decode center y-coordinate from predictions."""
    return predictions[:, 1:2] * priors[:, 3:4] * variances[1] + priors[:, 1:2]


def decode_width(predictions, priors, variances):
    """Decode width from predictions."""
    return priors[:, 2:3] * jp.exp(predictions[:, 2:3] * variances[2])


def decode_height(predictions, priors, variances):
    """Decode height from predictions."""
    return priors[:, 3:4] * jp.exp(predictions[:, 3:4] * variances[3])


def compute_boxes_center(predictions, priors, variances):
    """Compute center-form boxes from predictions."""
    center_x = decode_center_x(predictions, priors, variances)
    center_y = decode_center_y(predictions, priors, variances)
    W = decode_width(predictions, priors, variances)
    H = decode_height(predictions, priors, variances)
    return jp.concatenate([center_x, center_y, W, H], axis=1)


def convert_to_corner(boxes_center):
    """Convert center-form boxes to corner form."""
    return to_corner_form(boxes_center)


def combine_with_extras(boxes_corner, predictions):
    """Combine boxes with extra prediction data."""
    return jp.concatenate([boxes_corner, predictions[:, 4:]], axis=1)


def decode(predictions, priors, variances=[0.1, 0.1, 0.2, 0.2]):
    """Decode predicted box parameters to actual coordinates.

    Args:
        predictions (array): Encoded box predictions
        priors (array): Prior boxes
        variances (list): Decoding variances

    Returns:
        array: Decoded boxes in corner format with extras
    """
    boxes_center = compute_boxes_center(predictions, priors, variances)
    boxes_corner = convert_to_corner(boxes_center)
    return combine_with_extras(boxes_corner, predictions)


def get_best_box_coords(best_idx, x_minimum, x_maximum, y_minimum, y_maximum):
    """Retrieve coordinates of the best box."""
    best_xmin = x_minimum[best_idx]
    best_ymin = y_minimum[best_idx]
    best_xmax = x_maximum[best_idx]
    best_ymax = y_maximum[best_idx]
    return best_xmin, best_ymin, best_xmax, best_ymax


def compute_intersection_coords(x_minimum, y_minimum, x_maximum, y_maximum, best_coords):
    """Compute intersection coordinates between boxes."""
    best_xmin, best_ymin, best_xmax, best_ymax = best_coords
    return (
        jp.maximum(x_minimum, best_xmin),
        jp.maximum(y_minimum, best_ymin),
        jp.minimum(x_maximum, best_xmax),
        jp.minimum(y_maximum, best_ymax),
    )


def compute_intersection_area(intersection_coords):
    """Calculate the area of intersection between boxes."""
    xmin, ymin, xmax, ymax = intersection_coords
    w = jp.maximum(xmax - xmin, 0.0)
    h = jp.maximum(ymax - ymin, 0.0)
    return w * h


def compute_union(areas, best_idx, intersection):
    """Compute union area of two boxes."""
    return areas + areas[best_idx] - intersection


def calculate_IoU_with_best_box(x_minimum, x_maximum, y_minimum, y_maximum, best_idx, areas):
    """Calculate IoU between boxes and the best box."""
    best_coords = get_best_box_coords(best_idx, x_minimum, x_maximum, y_minimum, y_maximum)
    intersection_coords = compute_intersection_coords(x_minimum, y_minimum, x_maximum, y_maximum, best_coords)
    intersection = compute_intersection_area(intersection_coords)
    union = compute_union(areas, best_idx, intersection)
    return intersection / union


def _suppress_overlapping_boxes(best_idx, IoU, current_scores, IoU_thresh=0.45):
    """Suppress overlapping boxes by setting scores to -inf."""
    suppressed_scores = jp.where(IoU > IoU_thresh, -jp.inf, current_scores)
    indices = jp.arange(suppressed_scores.shape[0])
    mask = indices == best_idx
    updated_scores = jp.where(mask, -jp.inf, suppressed_scores)
    return updated_scores


def get_best_idx(scores):
    """Get index of the highest score."""
    return jp.argmax(scores).astype(jp.int32)


def update_indices(indices, i, best_idx):
    """Update indices array with best index."""
    idx = jp.arange(indices.shape[0])
    updated_indices = jp.where(idx == i, best_idx, indices)
    return updated_indices


def apply_NMS_iteration(
    x_minimum, y_minimum, x_maximum, y_maximum, areas, IoU_thresh, scores, index, indices
):
    """Perform one iteration of NMS processing."""
    best_idx = get_best_idx(scores)
    indices = update_indices(indices, index, best_idx)
    IoU = calculate_IoU_with_best_box(x_minimum, x_maximum, y_minimum, y_maximum, best_idx, areas)
    scores = _suppress_overlapping_boxes(best_idx, IoU, scores, IoU_thresh)
    return (index + 1, scores, indices)


def create_step_function(x_minimum, y_minimum, x_maximum, y_maximum, areas, IoU_thresh):
    """Create step function for NMS loop."""

    def step(state):
        index, scores, indices = state
        return apply_NMS_iteration(
            x_minimum, y_minimum, x_maximum, y_maximum, areas, IoU_thresh, scores, index, indices
        )

    return step


def handle_empty_case(boxes, scores):
    """Handle cases with no boxes."""
    if boxes.shape[0] == 0:
        return jp.zeros(0, dtype=jp.int32), 0
    return None


def compute_geometric_features(boxes):
    """Compute geometric features (x_min, etc.) from boxes."""
    x_minimum = boxes[:, 0]
    y_minimum = boxes[:, 1]
    x_maximum = boxes[:, 2]
    y_maximum = boxes[:, 3]
    areas = (x_maximum - x_minimum) * (y_maximum - y_minimum)
    return x_minimum, y_minimum, x_maximum, y_maximum, areas


def initialize_NMS_state(scores, top_k):
    """Initialize state for NMS processing."""
    selected_indices = jp.zeros(top_k, dtype=jp.int32)
    return (0, scores, selected_indices)


def create_condition_function(top_k):
    """Create condition function for NMS loop."""
    return lambda state: (state[0] < top_k) & (state[1].max() != -jp.inf)


def run_NMS_loop(condition_fn, step_fn, init_state):
    """Run the NMS loop using lax.while_loop."""
    return lax.while_loop(condition_fn, step_fn, init_state)


def process_final_results(final_state, top_k):
    """Process final NMS results to get selected indices."""
    count, _, indices = final_state
    num_selected = jp.minimum(count, top_k)
    selected = lax.dynamic_slice(indices, (0,), (num_selected,))
    return selected, num_selected


def compute_features_and_state(boxes, scores, top_k):
    """Compute features and initial state for NMS."""
    geometric_features = compute_geometric_features(boxes)
    init_state = initialize_NMS_state(scores, top_k)
    return geometric_features, init_state


def create_NMS_functions(geometric_features, IoU_thresh, top_k):
    """Create NMS functions for step and condition."""
    step_fn = create_step_function(*geometric_features, IoU_thresh)
    condition_fn = create_condition_function(top_k)
    return step_fn, condition_fn


def run_and_process(condition_fn, step_fn, init_state, top_k):
    """Run NMS loop and process results."""
    final_state = run_NMS_loop(condition_fn, step_fn, init_state)
    return process_final_results(final_state, top_k)


def execute_NMS_pipeline(boxes, scores, IoU_thresh, top_k):
    """Execute the NMS pipeline steps."""
    geometric_features, init_state = compute_features_and_state(boxes, scores, top_k)
    step_fn, condition_fn = create_NMS_functions(geometric_features, IoU_thresh, top_k)
    return run_and_process(condition_fn, step_fn, init_state, top_k)


def apply_non_max_suppression(boxes, scores, IoU_thresh=0.45, top_k=200):
    """Apply Non-Maximum Suppression to filter overlapping boxes.

    Args:
        boxes (array): Bounding boxes in corner format
        scores (array): Confidence scores for each box
        IoU_thresh (float): IoU threshold for suppression
        top_k (int): Maximum number of boxes to keep

    Returns:
        tuple: Selected indices and count of kept boxes
    """
    empty_result = handle_empty_case(boxes, scores)
    if empty_result is not None:
        return empty_result
    return execute_NMS_pipeline(boxes, scores, IoU_thresh, top_k)


def filter_confidence(boxes, scores, thresh):
    """Filter boxes below a confidence threshold."""
    mask = scores >= thresh
    return boxes[mask], scores[mask]


def get_NMS_indices(boxes, scores, NMS_thresh, top_k):
    """Get indices after NMS processing."""
    indices, count = apply_non_max_suppression(boxes, scores, NMS_thresh, top_k)
    return indices, count


def prepare_selections(boxes, scores, indices, count):
    """Prepare selected boxes and scores for output."""
    selected_boxes = boxes[indices[:count]]
    selected_scores = jp.expand_dims(scores[indices[:count]], -1)
    return jp.concatenate((selected_boxes, selected_scores), axis=1)


def filter_boxes(decoded_boxes, class_scores, confidence_thresh):
    mask = class_scores >= confidence_thresh
    scores = class_scores[mask]
    boxes = decoded_boxes[mask]
    return (boxes, scores) if len(scores) else None


def apply_nms_and_prepare(boxes, scores, NMS_thresh, top_k):
    indices, count = get_NMS_indices(boxes, scores, NMS_thresh, top_k)
    return prepare_selections(boxes, scores, indices, count)


def process_class(decoded_boxes, class_scores, NMS_thresh, confidence_thresh, top_k):
    filtered = filter_boxes(decoded_boxes, class_scores, confidence_thresh)
    if filtered is None:
        return None
    boxes, scores = filtered
    return apply_nms_and_prepare(boxes, scores, NMS_thresh, top_k)


def pad_selection(selection, top_k):
    """Pad selected boxes to fixed size."""
    pad_len = top_k - selection.shape[0]
    return jp.concatenate([selection, jp.zeros((pad_len, 5))], axis=0) if pad_len > 0 else selection[:top_k]


def process_class_output(class_arg, decoded_boxes, class_predictions, NMS_thresh, confidence_thresh, top_k):
    """Process class predictions and return padded outputs."""
    if class_arg == 0:
        return jp.zeros((top_k, 5))
    selection = process_class(
        decoded_boxes, class_predictions[:, class_arg], NMS_thresh, confidence_thresh, top_k
    )
    return jp.zeros((top_k, 5)) if selection is None else pad_selection(selection, top_k)


def decode_box_data(box_data):
    """Split box data into boxes and class predictions."""
    decoded_boxes = box_data[:, :4]
    class_predictions = box_data[:, 4:]
    return decoded_boxes, class_predictions


def create_outputs(num_classes, decoded_boxes, class_predictions, NMS_thresh, confidence_thresh, top_k):
    """Create outputs for each class after NMS."""
    outputs = []
    for i in range(num_classes):
        output = process_class_output(
            i, decoded_boxes, class_predictions, NMS_thresh, confidence_thresh, top_k
        )
        outputs.append(output)
    return outputs


def NMS_per_class(box_data, NMS_thresh=0.45, confidence_thresh=0.01, top_k=200):
    """Apply NMS per class to decoded box data.

    Args:
        box_data (array): Decoded box predictions with class scores
        NMS_thresh (float): IoU threshold for NMS
        confidence_thresh (float): Minimum confidence score
        top_k (int): Maximum detections per class

    Returns:
        array: Selected boxes and scores per class
    """
    decoded_boxes, class_predictions = decode_box_data(box_data)
    num_classes = class_predictions.shape[1]
    outputs = create_outputs(
        num_classes, decoded_boxes, class_predictions, NMS_thresh, confidence_thresh, top_k
    )
    return jp.stack(outputs, axis=0)


def get_boxes_coordinates_to_compute_IoU(box, boxes):
    """Get coordinates for IoU computation between a box and others."""
    x_minimum_A, y_minimum_A, x_maximum_A, y_maximum_A = box[:4]
    x_minimum_B, y_minimum_B = boxes[:, 0], boxes[:, 1]
    x_maximum_B, y_maximum_B = boxes[:, 2], boxes[:, 3]
    return (x_minimum_A, y_minimum_A, x_maximum_A, y_maximum_A,
            x_minimum_B, y_minimum_B, x_maximum_B, y_maximum_B,)


def get_inner_coordinates(
    x_minimum_A, y_minimum_A, x_maximum_A, y_maximum_A, x_minimum_B, y_minimum_B, x_maximum_B, y_maximum_B
):
    """Compute inner coordinates for intersection area."""
    inner_x_minimum = jp.maximum(x_minimum_B, x_minimum_A)
    inner_y_minimum = jp.maximum(y_minimum_B, y_minimum_A)
    inner_x_maximum = jp.minimum(x_maximum_B, x_maximum_A)
    inner_y_maximum = jp.minimum(y_maximum_B, y_maximum_A)
    return inner_x_minimum, inner_y_minimum, inner_x_maximum, inner_y_maximum


def calculate_intersection(inner_x_minimum, inner_y_minimum, inner_x_maximum, inner_y_maximum):
    """Calculate intersection area between two boxes."""
    inner_w = jp.maximum((inner_x_maximum - inner_x_minimum), 0)
    inner_h = jp.maximum((inner_y_maximum - inner_y_minimum), 0)
    return inner_w * inner_h


def calculate_union(x_minimum_A, y_minimum_A, x_maximum_A, y_maximum_A, x_minimum_B,
                    y_minimum_B, x_maximum_B, y_maximum_B, intersection_area):
    """Calculate union area of two boxes."""
    box_area_B = (x_maximum_B - x_minimum_B) * (y_maximum_B - y_minimum_B)
    box_area_A = (x_maximum_A - x_minimum_A) * (y_maximum_A - y_minimum_A)
    union_area = box_area_A + box_area_B - intersection_area
    return union_area


def compute_iou(box, boxes):
    """Compute Intersection over Union (IoU) between a box and multiple boxes.

    Args:
        box (array): Single box coordinates [x_min, y_min, x_max, y_max]
        boxes (array): Array of boxes in same format

    Returns:
        array: IoU values for each box in 'boxes'
    """
    coordinates = get_boxes_coordinates_to_compute_IoU(box, boxes)
    inner_coordinates = get_inner_coordinates(*coordinates)
    intersection_area = calculate_intersection(*inner_coordinates)
    union_area = calculate_union(*coordinates, intersection_area)
    return intersection_area / union_area
