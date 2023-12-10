import torch
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
import easyocr
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64
import os
import pytesseract

class FormOCR():
    def analyze(self, filename):
        # predict the boundary box
        boxes, scores = self.predict_box(filename)

        # draw the predicted box on the image
        output_image = self.draw_box(filename, boxes)

        # crop the image based on the box predicted
        self.crop_image(filename, tuple(boxes[0].tolist()))

        # perform ocr on cropped section
        extracted_text = self.run_ocr(filename)
        # extracted_text = self.alternate_run_ocr(filename)

        return  {
            "image": output_image,
            "text": extracted_text,
            "score": scores
        }

    def predict_box(self, filename):
        # load model
        model = torch.load('models/findbox.pth')
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.eval()
        original_img = Image.open('uploads/' + filename).convert("L")
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img = transform(original_img).unsqueeze(0).to(device)

        # apply uploaded image into the boundary prediction model

        output = model(img)
        boxes = output[0]['boxes'].detach().to(device)
        scores = output[0]['scores'].detach().to(device)

        # find the box with the highest condifence score
        max_score_idx = torch.argmax(scores)
        boxes = boxes[max_score_idx].unsqueeze(0)
        scores = scores[max_score_idx].unsqueeze(0).item()

        return boxes, scores

    def draw_box(self, filename, box):
        original_img = Image.open('uploads/' + filename).convert("L")

        # Convert image into Tensor of shape (C x H x W) and dtype uint8
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])
        img_tensor = transform(original_img)

        # draw boundary on the image
        toPilTransform = transforms.Compose([
            transforms.ToPILImage()
        ])

        output_image = toPilTransform(draw_bounding_boxes(
            image=img_tensor, boxes=box, colors=(255, 0, 0), width=3))

        # convert image into base64 string
        image_buffer = BytesIO()
        output_image.save(image_buffer, format="PNG")
        base64_image = base64.b64encode(image_buffer.getvalue()).decode("utf-8")

        return base64_image

    def crop_image(self, filename, box):
        img = Image.open('uploads/' + filename).convert("L")
        img = img.crop(box)
        img.save('uploads/' + filename[:-4] + '_cropped.png')

    def run_ocr(self, filename):
        reader = easyocr.Reader(['en'])
        text = reader.readtext('uploads/' + filename[:-4] + '_cropped.png')
        text_list = []
        for t in text:
            t = (t[1], round(t[2], 2))
            text_list.append(t)
        print(text_list)
        prompt = 'Extract full name and address from the easyocr result into json object named info.\n ' + \
            ', '.join([f"('{item[0]}', {item[1]})" for item in text_list])
        data = {'content': prompt, 'role': "user"}
        chatgpt = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = chatgpt.chat.completions.create(
            model='gpt-3.5-turbo-1106',
            response_format={"type": "json_object"},
            messages=[data],
            temperature=0
        )
        return (response.choices[0].message.content)


    def alternate_run_ocr(self, filename):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Replace with the actual path on your system
        img = Image.open('uploads/' + filename[:-4] + '_cropped.png')
        text = pytesseract.image_to_string(img)
        print(text)

