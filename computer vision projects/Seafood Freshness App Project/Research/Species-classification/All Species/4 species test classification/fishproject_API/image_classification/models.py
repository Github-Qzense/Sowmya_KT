from django.db import models

# Create your models here.

class Image(models.Model):
    image = models.ImageField(upload_to='media/')
    # Add any other fields you need for your model
