import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import os
import json

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.upconv3(b)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.out(d1))

class CrackDetectorNN:
    def __init__(self, model_path='crack_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet(in_channels=1, out_channels=1).to(self.device)
        self.model_path = model_path
        self.trained = False

        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                self.trained = True
                print(f"Загружена обученная модель из {model_path}")
            except Exception as e:
                print(f"Не удалось загрузить модель: {e}, используется новая")

        self.model.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

    def preprocess(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Не удалось загрузить изображение")

        original_size = img.shape
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img)
        h, w = img_enhanced.shape
        img_resized = cv2.resize(img_enhanced, (256, 256))
        img_tensor = torch.from_numpy(img_resized).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 256, 256)

        return img_tensor.to(self.device), original_size, img_resized

    def predict(self, image_path):
        self.model.eval()

        with torch.no_grad():
            img_tensor, original_size, img_resized = self.preprocess(image_path)
            output = self.model(img_tensor)
            prob_mask = output.squeeze().cpu().numpy()

        return prob_mask, img_resized, original_size

    def train_on_feedback(self, image_path, bboxes, labels):
        self.model.train()

        img_tensor, _, _ = self.preprocess(image_path)
        target_mask = torch.zeros((1, 1, 256, 256), device=self.device)

        for bbox, is_crack in zip(bboxes, labels):
            if is_crack:
                x, y, w, h = bbox
                orig_h, orig_w = img_tensor.shape[2], img_tensor.shape[3]
                x_scaled = int(x / orig_w * 256)
                y_scaled = int(y / orig_h * 256)
                w_scaled = max(2, int(w / orig_w * 256))
                h_scaled = max(2, int(h / orig_h * 256))
                x1 = min(255, max(0, x_scaled))
                y1 = min(255, max(0, y_scaled))
                x2 = min(255, max(0, x_scaled + w_scaled))
                y2 = min(255, max(0, y_scaled + h_scaled))

                if x2 > x1 and y2 > y1:
                    target_mask[0, 0, y1:y2, x1:x2] = 1.0

        losses = []
        for epoch in range(15):
            self.optimizer.zero_grad()
            output = self.model(img_tensor)
            loss = self.criterion(output, target_mask)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        torch.save(self.model.state_dict(), self.model_path)
        self.model.eval()
        self.trained = True
        avg_loss = sum(losses) / len(losses)
        print(f"Обучение завершено. Средняя потеря: {avg_loss:.4f}")
        return avg_loss

def classical_crack_detection(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Не удалось загрузить изображение")

    original = img.copy()
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, body_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours_body, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_body:
        largest_contour = max(contours_body, key=cv2.contourArea)
        body_mask = np.zeros_like(gray)
        cv2.drawContours(body_mask, [largest_contour], -1, 255, -1)

    body_pixels = gray[body_mask > 0]
    if len(body_pixels) > 0:
        bone_threshold = np.percentile(body_pixels, 65)
    else:
        bone_threshold = 160

    _, bone_mask = cv2.threshold(gray, bone_threshold, 255, cv2.THRESH_BINARY)
    bone_mask = cv2.bitwise_and(bone_mask, body_mask)

    kernel_bone = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_CLOSE, kernel_bone)
    enhanced_bone = enhanced.copy()
    enhanced_bone[bone_mask == 0] = 0
    mean_bone = np.mean(enhanced_bone[bone_mask > 0]) if np.any(bone_mask > 0) else 128

    _, binary_global = cv2.threshold(enhanced_bone, mean_bone * 0.7, 255, cv2.THRESH_BINARY_INV)
    binary_adaptive = cv2.adaptiveThreshold(enhanced_bone, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                            15, 3)
    binary = cv2.bitwise_or(binary_global, binary_adaptive)
    binary = cv2.bitwise_and(binary, bone_mask)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_line = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_line)

    edges = cv2.Canny(enhanced_bone, 20, 80)
    edges = cv2.bitwise_and(edges, bone_mask)
    combined = cv2.bitwise_or(binary, edges)

    lines = cv2.HoughLinesP(combined, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=10)
    line_mask = np.zeros_like(gray)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
    line_mask = cv2.bitwise_and(line_mask, bone_mask)

    final_mask = cv2.bitwise_or(combined, line_mask)

    return final_mask, bone_mask, original, enhanced, width, height

crack_detector = None


def load_crack_model(weights_path=None):
    global crack_detector
    if weights_path:
        crack_detector = CrackDetectorNN(model_path=weights_path)
    else:
        crack_detector = CrackDetectorNN()
    return crack_detector


def analyze_xray(image_path):
    global crack_detector

    if crack_detector is None:
        crack_detector = CrackDetectorNN()

    final_mask, bone_mask, original, enhanced, width, height = classical_crack_detection(image_path)

    if crack_detector.trained:
        try:
            prob_mask, img_resized, original_size = crack_detector.predict(image_path)

            nn_mask = (prob_mask > 0.25).astype(np.uint8) * 255
            nn_mask = cv2.resize(nn_mask, (width, height))

            final_mask = cv2.bitwise_or(final_mask, nn_mask)
        except Exception as e:
            print(f"Ошибка нейросети, используем только CV: {e}")

    output = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]

    findings = []
    annotated = original.copy()
    min_area = 15
    max_area = (width * height) * 0.05

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area or area > max_area:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        region_mask = (labels == i).astype(np.uint8)
        overlap = cv2.countNonZero(cv2.bitwise_and(region_mask, bone_mask))
        if overlap < area * 0.3:
            continue

        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        box_w, box_h = rect[1]
        if box_w < 1 or box_h < 1:
            continue

        aspect_ratio = max(box_w, box_h) / min(box_w, box_h)

        region_intensity = np.mean(enhanced[region_mask == 1])
        bone_surrounding = cv2.bitwise_and(bone_mask, cv2.bitwise_not(region_mask))
        if cv2.countNonZero(bone_surrounding) > 0:
            bg_intensity = np.mean(enhanced[bone_surrounding > 0])
        else:
            bg_intensity = 200
        intensity_ratio = region_intensity / (bg_intensity + 1)

        score = 0.0

        if aspect_ratio > 4:
            score += 0.35
        elif aspect_ratio > 2.5:
            score += 0.25
        elif aspect_ratio > 1.8:
            score += 0.15
        if intensity_ratio < 0.7:
            score += 0.35
        elif intensity_ratio < 0.85:
            score += 0.25
        elif intensity_ratio < 0.95:
            score += 0.15

        probability = min(score + 0.1, 0.95)
        probability = max(probability, 0.08)

        if probability > 0.6:
            color = (0, 0, 255)
            thickness = 3
            crack_type = "вероятная трещина"
        elif probability > 0.35:
            color = (0, 140, 255)
            thickness = 2
            crack_type = "возможная микротрещина"
        else:
            color = (0, 255, 200)
            thickness = 2
            crack_type = "зона интереса"

        padding = 4
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        label = f"{probability * 100:.0f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 2)
        text_x = x1
        text_y = y1 - 10 if y1 > 30 else y2 + text_h + 10

        cv2.rectangle(annotated, (text_x - 2, text_y - text_h - 4), (text_x + text_w + 2, text_y + 2), (0, 0, 0), -1)
        cv2.putText(annotated, label, (text_x, text_y), font, font_scale, (255, 255, 255), 2)

        findings.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "mean_probability": round(probability, 4),
            "crack_type": crack_type,
            "area_pixels": int(area)
        })

    bone_contours, _ = cv2.findContours(bone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, bone_contours, -1, (255, 150, 0), 1)

    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (width, 45), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    if len(findings) > 0:
        info_text = f"Found: {len(findings)} suspicious zones"
        text_color = (0, 255, 200)
    else:
        info_text = "No cracks detected"
        text_color = (0, 255, 100)

    cv2.putText(annotated, info_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    result_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    return result_img, findings


def train_model_from_feedback(image_path, feedback_data):
    global crack_detector

    if crack_detector is None:
        crack_detector = CrackDetectorNN()

    bboxes = [f['bbox'] for f in feedback_data]
    labels = [f['doctor_confirmed'] for f in feedback_data]

    if any(labels):
        try:
            loss = crack_detector.train_on_feedback(image_path, bboxes, labels)
            print(f"Нейросеть обучена! Потеря: {loss:.4f}")
            return loss
        except Exception as e:
            print(f"Ошибка обучения: {e}")
            return None

    return None