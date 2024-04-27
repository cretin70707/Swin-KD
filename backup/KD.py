import torch.nn.functional as F
from tabnanny import verbose
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F
from torch.nn import functional as F
from backbones.swin_transformer import SwinTransformer3D
from einops import rearrange
from utils.tensor_utils import Reduce
from losses.mse import mse_loss
from losses.r2 import r2_loss
from sklearn.metrics import r2_score
from losses.rmse import RMSE

class KDFrame(pl.LightningModule):
    def __init__(self, teacher_model, student_model, temperature=1.0, alpha=0.5):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_model.eval()
        self.batch_size = 4
        
    def forward(self, nvideo):
        return self.student_model(nvideo)  

    def training_step(self, batch, batch_idx):
        filename, nvideo, nlabel, ejection, repeat, fps = batch
        ef_label = ejection.type(torch.float32) / 100.

        # Forward pass through the teacher and student models
        with torch.no_grad():
            teacher_outputs = self.teacher_model(nvideo)
        student_outputs = self.student_model(nvideo)

        # Calculate the distillation loss
        loss = self.KD_loss(teacher_outputs, student_outputs, ef_label)
        self.log('loss', loss, on_epoch=True, batch_size=self.batch_size)

        return loss
        
    def validation_step(self, batch, batch_idx):
        filename, nvideo, nlabel, ejection, repeat, fps = batch
        ef_label = ejection.type(torch.float32) / 100.
        ef_pred = self(nvideo)
        loss = F.mse_loss(ef_pred,ef_label)

        self.log('val_loss', loss, on_epoch=True, batch_size=self.batch_size, prog_bar=True)

        return {"val_loss":loss}

    def KD_loss(self, teacher_outputs, student_outputs, targets):

        student_loss = F.mse_loss(student_outputs, targets)
        # Compute the knowledge distillation loss
        teacher_outputs = teacher_outputs / self.temperature
        student_outputs = student_outputs / self.temperature
        
        soft_teacher = F.softmax(teacher_outputs, dim=-1)
        soft_student = F.log_softmax(student_outputs, dim=-1)
        distillation_loss = F.mse_loss(soft_student,soft_teacher) * (self.temperature ** 2)

        # Combine the losses with a weighting factor (alpha)
        total_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        return total_loss
    

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=1e-4, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85, verbose=True)

        return [optimizer], [lr_scheduler]