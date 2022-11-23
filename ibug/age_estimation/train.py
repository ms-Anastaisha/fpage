from argparse import Namespace
from pathlib import Path
import torch 
import torchvision.transforms as T
import yaml
from AgeDataset import AgeDataModule
from ibug.face_parsing.parser import FaceParser
from ibug.face_detection import RetinaFacePredictor
import ibug.roi_tanh_warping.reference_impl as ref
from torchvision.models._utils import IntermediateLayerGetter

from fpage import FPAge
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

def face_parse_forward(face_parser, image_batch, bboxes):
    input_shape = image_batch.shape[-2:]
    features = face_parser.encoder(image_batch, bboxes)
    out = face_parser.decoder(features["c4"])
    logits, high = out["logits"], out["high"]
    c2 = features["c2"]
    mask = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)
    
    return c2, high, logits, mask


def train(train_params, age_model, face_parser, device):
    epoch_idx = 0
    best_vloss = 1000000
    data_module = AgeDataModule(train_params)
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    save_dir = Path(train_params.save_dir)
    
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
    l1_loss = torch.nn.L1Loss()
    criterion = lambda input, target: kl_loss(input, target) + l1_loss(input, target)
    
    optimizer = torch.optim.Adam(age_model.parameters(), lr=train_params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, patience=train_params.patience)

    tb_writer = SummaryWriter(train_params.verbose_dir)

    age_model.to(device)

    for epoch in range(train_params.epochs):
        age_model.train(True)
        print(f"EPOCH {epoch_idx + 1}")
        train_loss = 0
        for idx, data in enumerate(train_dataloader):
            inputs, labels = data
            
            for img in inputs:
                bboxes = train_params.face_detector(img.numpy(), rgb=False)
                img  = ref.roi_tanh_polar_warp(img, bboxes[0], *train_params.size, keep_aspect_ratio=True)
                bboxes_tensor.append(torch.tensor(bboxes[0]).view(1, -1))
            bboxes_tensor = torch.stack(bboxes_tensor)
            bboxes = train_params.face_detector(img, rgb=False)
            img  = ref.roi_tanh_polar_warp(img, bboxes[0], *train_params.size, keep_aspect_ratio=True)
            bboxes_tensor.append(torch.tensor(bboxes[0]).view(1, -1))
            inputs, bboxes_tensor, labels  = (inputs.to(device), 
                                              bboxes_tensor.to(device), 
                                              labels.to(device))
            
            optimizer.zero_grad()
            low, high, logits, _ = face_parse_forward(face_parser, inputs, bboxes_tensor)
            outputs = age_model(low, high, logits)
        
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            tb_writer.add_scalar("Loss/train_loss_step", loss.item(), epoch*len(train_dataloader) + idx)
        
        age_model.train(False)
        val_loss = 0
        with torch.no_grad():
            for idx, data in enumerate(val_dataloader):
                inputs, bboxes_tensor, labels = data
                inputs, bboxes_tensor, labels = (inputs.to(device), 
                bboxes_tensor.to(device), labels.to(device))
                low, high, logits, _ = face_parse_forward(face_parser, inputs, bboxes_tensor)
                outputs = age_model(low, high, logits)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)
        print(f"LOSS train {train_loss/ len(train_dataloader)} val {val_loss/len(val_dataloader)}")
        tb_writer.add_scalars("Training vs Validation Loss", {"Training": train_loss, "Validation": val_loss})
        scheduler.step(val_loss)

    if val_loss < best_vloss:
        best_vloss = val_loss
        model_path = save_dir / f"model_{idx}"
        torch.save(age_model.state_dict(), str(model_path))

def test():
    ...

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("/home/user/client_projects/15secondsoffame/ai-features/fpage/ibug/age_estimation/train_conf.yml", "r") as f:
    train_params = Namespace(**yaml.safe_load(f))
train_params.preprocess = T.Compose([T.ToTensor(), T.Normalize(train_params.mean, train_params.std), 
                                    T.Resize(train_params.size)])


if __name__ == "__main__":
    age_model = FPAge(n_blocks=train_params.n_blocks, 
    age_classes=train_params.age_classes, face_classes=train_params.face_classes)
    face_parser = FaceParser(device=device, encoder='resnet50', decoder='fcn', 
    num_classes=train_params.face_classes).model
    face_parser.decoder = IntermediateLayerGetter(
                face_parser.decoder, {"2": "high", "4": "logits"}
            )
    face_detector =  RetinaFacePredictor(
        threshold=0.8,
        device=device,
        model=(RetinaFacePredictor.get_model("mobilenet0.25")),
    )
    train_params.face_detector = face_detector
    train(train_params, age_model, face_parser, device=device)
    test(train_params, age_model, face_parser, device=device)
    