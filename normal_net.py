import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import construct_backbone
from data.config import cfg,set_cfg
from utils import timer

torch.cuda.current_device()

class TestNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = construct_backbone(cfg.backbone)
        self.freeze_bn()
        self.normal_decoder = NormalDecoder()

    def forward(self, x):
        with timer.env("backbone"):
            features_encoder = self.backbone(x)
        with timer.env("decoder"):
            normal_pred = self.normal_decoder(features_encoder)
        return normal_pred

    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def init_weights(self, backbone_path):
        """ Initialize weights for training. """
        # Initialize the backbone with the pretrained weights.
        self.backbone.init_backbone(backbone_path)


        for name, module in self.named_modules():
            is_conv_layer = isinstance(module, nn.Conv2d)  # or is_script_conv
            if is_conv_layer and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)
                    
    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()
                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

class NormalDecoder(nn.Module):
    def __init__(self):
        super(NormalDecoder, self).__init__()
        self.num_output_channels = 3

        self.deconv1 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest', align_corners=None),
            nn.ReflectionPad2d(1),
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(1024, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest', align_corners=None),
            nn.ReflectionPad2d(1),
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest', align_corners=None),
            nn.ReflectionPad2d(1),
            nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest', align_corners=None),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        
        self.normal_pred = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, self.num_output_channels, kernel_size=3, stride=1, padding=0),
            nn.Tanh()
        )
        
    def forward(self, feature_maps):
        feats = list(reversed(feature_maps))
        
        x = self.deconv1(feats[0])
        x = self.deconv2(torch.cat([feats[1], x], dim=1))
        x = self.deconv3(torch.cat([feats[2], x], dim=1))
        x = self.deconv4(torch.cat([feats[3], x], dim=1))
        x = self.normal_pred(x)
        x = F.interpolate(x, scale_factor=2,align_corners=False, mode='bilinear')
        valid_mask = torch.pow(x, 2).sum(dim=1) > 1e-3
        x = x * valid_mask.unsqueeze(dim=1).repeat(1,3,1,1)
        x = F.normalize(x, p=2, dim=1)
        return x


if __name__ == "__main__":

    import argparse
    def parse_args(argv=None):
        parser = argparse.ArgumentParser(description="For PlaneRecNet Debugging and Inference Time Measurement")
        parser.add_argument(
            "--trained_model",
            default=None,
            type=str,
            help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.',
        )
        parser.add_argument(
            "--config", 
            default="PlaneRecNet_50_config", 
            help="The config object to use.")
        parser.add_argument(
            "--fps", 
            action="store_true", 
            help="Testing running speed.")
        global args
        args = parser.parse_args(argv)
    
    parse_args()
    from utils.utils import MovingAverage, init_console

    init_console()

    set_cfg('resnet50_dcnv2_backbone')
    net = TestNet(cfg)
    net.init_weights(backbone_path="weights/resnet50-19c8e357.pth")
    net = net.cuda()
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    batch = torch.zeros((1, 3, 512, 512)).cuda().float()

    y = net(batch)
    print(y.shape)

    if args.fps:
        net(batch)
        avg = MovingAverage()
        try:
            while True:
                timer.reset()
                with timer.env("everything else"):
                    net(batch)
                avg.add(timer.total_time())
                print("\033[2J")  # Moves console cursor to 0,0
                timer.print_stats()
                print(
                    "Avg fps: %.2f\tAvg ms: %.2f         "
                    % (1000 / avg.get_avg(), avg.get_avg())
                )
        except KeyboardInterrupt:
            pass
    else:
        exit()

   



