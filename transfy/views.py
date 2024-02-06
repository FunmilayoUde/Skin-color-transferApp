from django.shortcuts import render
from django.core.files.base import ContentFile
from django.http import HttpResponse
from .forms import ImageUploadForm
from .skinsegmentation import load_skin_segmentation_model,create_skin_mask, refine_mask
from .colordetection import extractSkin,extractDominantColor, adjust_hsv_dominance
from .transfer import applyAdjustedColorToSkinRegion, blendSkinWithTexture
import os
import numpy as np
import base64
import cv2
from PIL import Image
from django.conf import settings  # Import settings for BASE_DIR


# Create your views here.

model_path = 'transfy/model_segmentation_realtime_skin_30.pth'


def home_page(request):
    return render(request, 'index.html')  # Assuming your template is named 'index.html'


def process_view(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            
            # Access uploaded images using form.cleaned_data['source_image'] and form.cleaned_data['target_image']
            source_image_file = form.cleaned_data['source_image']
            target_image_file = form.cleaned_data['target_image']


            # Assuming source_image_file is an InMemoryUploadedFile or TemporaryUploadedFile
            source_image = Image.open(source_image_file)
            source_image = np.array(source_image)

            # Similarly for target_image
            target_image = Image.open(target_image_file)
            target_image = np.array(target_image)


            # Perform your image processing using your existing scripts
            model = load_skin_segmentation_model(model_path)
            print(target_image.shape)
            skin_mask_source, resized_source = create_skin_mask(model,source_image)
            skin_mask_target, resized_target = create_skin_mask(model,target_image)

            refined_source_mask = refine_mask(resized_source, model, skin_mask_source)
            refined_target_mask = refine_mask(resized_target, model, skin_mask_target)

            source_skin = extractSkin(resized_source, refined_source_mask)
            target_skin = extractSkin(resized_target, refined_target_mask)

            dominant_color = extractDominantColor(cv2.cvtColor(source_skin, cv2.COLOR_BGR2RGB), number_of_colors = 5, hasThresholding = True)
            adjusted_color = adjust_hsv_dominance(dominant_color, hsv_adjust = 1)

            
            target_skin_result = applyAdjustedColorToSkinRegion(target_skin, refined_target_mask, adjusted_color)
            # Generate the final image
            processed_image = blendSkinWithTexture(target_skin,refined_target_mask,target_skin_result, resized_target)
            
            processed_image_file = ContentFile(cv2.imencode('.jpg', processed_image)[1].tobytes())
            base64_image_string = base64.b64encode(processed_image_file.read()).decode('utf-8')
            print(base64_image_string)
            
            context = {
                'source_form': form,
                'base64_image': base64_image_string
            }
            return render(request, 'index.html', context)
    else:
        form = ImageUploadForm()
        context = {'form': form}
        return render(request, 'index.html', context)


