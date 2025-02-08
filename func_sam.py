import os
import cv2
import torch
import numpy as np

from PIL import Image
from typing import Any, Dict, List
from transformers import (
    pipeline,
    AutoProcessor,
    AutoModelForMaskGeneration,
)

class SAM:
    def __init__(self, detect_id: str = None, segment_id: str = None, device: str = 'cuda'):
        self.detect_id = detect_id or 'IDEA-Research/grounding-dino-tiny'
        self.detect_model = None
        self.segment_id = segment_id or 'facebook/sam-vit-huge'
        self.segment_model = None
        self.segment_processor = None
        self.device = device

    def load(self):
        self.detect_model = pipeline(model=self.detect_id, task="zero-shot-object-detection", device=self.device)
        self.segment_model = AutoModelForMaskGeneration.from_pretrained(self.segment_id).to(self.device)
        self.segment_processor = AutoProcessor.from_pretrained(self.segment_id)

    def detect(self, image: Image.Image, labels: List[str], threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
        """
        candidate_labels = []
        for label in labels:
            label = label.strip()
            candidate_labels.append(label if label.endswith('.') else label + '.')
        
        results = self.detect_model(image,candidate_labels=candidate_labels,threshold=threshold)
        return results

    def refine_masks(self, masks: List) -> List:
        masks = masks.cpu().float()
        masks = masks.permute(0, 2, 3, 1)
        masks = masks.mean(axis=-1)
        masks = (masks > 0).int()
        masks = masks.numpy().astype(np.uint8)
        masks = list(masks)
        result = []
        for mask in masks:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            polygon = largest_contour.reshape(-1, 2).tolist()
            mask = np.zeros(mask.shape, dtype=np.uint8)
            pts = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask, [pts], color=(255,))
            result.append(mask)
        return result

    def segment(self, image: Image.Image, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        boxes = [[list(item['box'].values()) for item in detections]]
        inputs = self.segment_processor(images=image, input_boxes=boxes, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.segment_model(**inputs)

        masks = self.segment_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]

        masks = self.refine_masks(masks)
        for detection, mask in zip(detections, masks):
            # detection['label'] = str(detection['label']).replace('.', '')
            detection['mask'] = mask
        return detections

    def annotate(self, image: Image.Image, detections: List[Dict]) -> np.ndarray:
        # Convert PIL Image to OpenCV format
        image_cv2 = np.array(image)
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

        # Iterate over detections and add bounding boxes and masks
        for detection in detections:
            label = detection['label']
            score = detection['score']
            box = detection['box']
            mask = detection['mask']

            # Sample a random color for each detection
            color = np.random.randint(0, 256, size=3).tolist()

            # Draw bounding box
            pt1 = (box['xmin'], box['ymin'])
            pt2 = (box['xmax'], box['ymax'])
            cv2.rectangle(image_cv2, pt1, pt2, color, 2)
            cv2.putText(image_cv2, f'{label}: {score:.2f}', (box['xmin'], box['ymin'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # If mask is available, apply it
            if mask is not None:
                # Convert mask to uint8
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image_cv2, contours, -1, color, 2)
        return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    def __call__(self, image: Image.Image, labels: str, threshold: float = 0.3, **kwargs):
        labels = labels.strip().split(',')
        detections = self.detect(image, labels, threshold)
        detections = self.segment(image, detections)
        masks = {}
        for detection in detections:
            mask = Image.fromarray(detection['mask']).convert('L')
            label = detection['label'].replace('.', '').strip()
            old_mask = masks.get(label, Image.new('RGB', mask.size, (0, 0, 0)))
            mask = Image.composite(mask, old_mask, mask)
            masks[label] = mask
        annotated = Image.fromarray(self.annotate(image, detections))
        return masks, annotated
