
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        # GroupNorm is often better than BatchNorm for small batch sizes (which CPU requires)
        self.gn1 = nn.GroupNorm(num_groups=min(num_groups, out_c), num_channels=out_c)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=min(num_groups, out_c), num_channels=out_c)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu2(x)
        return x

class RibUNet(nn.Module):
    """
    Lightweight 3D U-Net for Rib Segmentation.
    Optimized for CPU training (low parameter count).
    Structure: Encoders -> Bottleneck -> Decoders
    Depth: 3 layers (16 -> 32 -> 64).
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=16, apply_sigmoid: bool = False):
        super().__init__()
        self.apply_sigmoid = bool(apply_sigmoid)
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_filters)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = ConvBlock(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_filters * 2, base_filters * 4)
        
        # Decoder
        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_filters * 4, base_filters * 2) # Concat inputs: 2x + 2x = 4x channels in
        
        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_filters * 2, base_filters) # Concat inputs: 1x + 1x = 2x channels in
        
        # Final Output
        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)
        
    def forward(self, x, apply_sigmoid=None):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder
        u2 = self.up2(b)
        # Center crop if needed? Assuming padding=same, so shapes usually match if input is multiple of 16
        # Torch cat along channel dim (1)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        logits = self.final_conv(d1)
        use_sigmoid = self.apply_sigmoid if apply_sigmoid is None else bool(apply_sigmoid)
        if use_sigmoid:
            return torch.sigmoid(logits)
        return logits

    def predict_proba(self, x):
        """Computes voxel-wise centerline probabilities in [0, 1] via sigmoid(logits)."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x, apply_sigmoid=False))

    @classmethod
    def load(cls, filepath, map_location="cpu"):
        state_dict = torch.load(filepath, map_location=map_location)
        in_c = state_dict["enc1.conv1.weight"].shape[1] if "enc1.conv1.weight" in state_dict else 1
        base_f = state_dict["enc1.conv1.weight"].shape[0] if "enc1.conv1.weight" in state_dict else 16
        out_c = state_dict["final_conv.weight"].shape[0] if "final_conv.weight" in state_dict else 1
        model = cls(in_channels=in_c, out_channels=out_c, base_filters=base_f)
        model.load_state_dict(state_dict)
        model.eval()
        return model


RibUNet3D = RibUNet

