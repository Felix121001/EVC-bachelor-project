import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.nn.functional as F

from GAN.transformer_torch import TransformerBlock


from GAN.utils import (
    DownSample,
    upSample,
    upSample1D,
    ResidualLayer,
    ResidualLayer_combined,
    conv_block_gated,
    conv_block_gated_1D,
    conv_block,
    encoder_block,
    encoder_block_1D,
    decoder_block,
    decoder_block_1D,
    visualize_f0,
    visualize_coded_spectrogram,
)


class build_unet(nn.Module):
    def __init__(self, gated=False):
        super().__init__()
        """ Encoder """
        self.gated = gated
        self.e1 = encoder_block(1, 64, self.gated)
        self.e2 = encoder_block(64, 128, self.gated)
        self.e3 = encoder_block(128, 256, self.gated)
        self.e4 = encoder_block(256, 512, self.gated)
        """ Bottleneck """
        self.b = conv_block_gated(512, 1024) if self.gated else conv_block(512, 1024)
        """ Decoder """
        self.d1 = decoder_block(1024, 512, gated=self.gated)
        self.d2 = decoder_block(512, 256, gated=self.gated)
        self.d3 = decoder_block(256, 128, gated=self.gated)
        self.d4 = decoder_block(128, 64, gated=self.gated)
        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        """ Encoder """
        print("inputs.shape", inputs.shape)
        s1, p1 = self.e1(inputs)
        print("p1.shape", p1.shape, "s1.shape", s1.shape)
        s2, p2 = self.e2(p1)
        print("p2.shape", p2.shape, "s2.shape", s2.shape)
        s3, p3 = self.e3(p2)
        print("p3.shape", p3.shape, "s3.shape", s3.shape)
        s4, p4 = self.e4(p3)
        print("p4.shape", p4.shape, "s4.shape", s4.shape)
        """ Bottleneck """
        b = self.b(p4)
        print("b.shape", b.shape)
        """ Decoder """
        d1 = self.d1(b, s4)
        print("d1.shape", d1.shape)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4).squeeze(1)
        print("outputs.shape", outputs.shape)
        return outputs


class build_unet_residual(build_unet):
    def __init__(self, gated=True):
        super().__init__(gated)

        self.residualLayer1 = ResidualLayer(
            in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1
        )
        self.residualLayer2 = ResidualLayer(
            in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1
        )
        self.residualLayer3 = ResidualLayer(
            in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1
        )

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        """ Encoder """
        # print("inputs.shape", inputs.shape)
        s1, p1 = self.e1(inputs)
        # print("p1.shape", p1.shape, "s1.shape", s1.shape)
        s2, p2 = self.e2(p1)
        # print("p2.shape", p2.shape, "s2.shape", s2.shape)
        s3, p3 = self.e3(p2)
        # print("p3.shape", p3.shape, "s3.shape", s3.shape)
        s4, p4 = self.e4(p3)
        # print("p4.shape", p4.shape, "s4.shape", s4.shape)
        """ Bottleneck """
        residual_layer_1 = self.residualLayer1(p4)
        residual_layer_2 = self.residualLayer2(residual_layer_1)
        # residual_layer_3 = self.residualLayer3(residual_layer_2)
        b = self.b(residual_layer_2)
        # print("b.shape", b.shape)
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4).squeeze(1)
        # print("outputs.shape", outputs.shape)
        return outputs


class ClassConditioning(nn.Module):
    def __init__(self, height, width, inp, num_channels=None):
        super(ClassConditioning, self).__init__()

        self.num_channels = num_channels if num_channels is not None else inp
        self.height = height
        self.width = width
        self.linear = nn.Linear(
            in_features=inp, out_features=height * width * self.num_channels
        )

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = x.view(
            -1, self.num_channels, self.height, self.width
        )  # Assuming input x is of shape (batch_size, inp)
        return x


class DualPathwayModel(nn.Module):
    def __init__(self, csv):
        super().__init__()
        self.gated = csv["MODEL"]["GATED_CONV"]
        self.num_mcep = csv["INPUT"]["NUM_MCEP"]
        self.num_classes = len(csv["DATASET"]["EMOTIONS"])
        self.n_frames = csv["INPUT"]["N_FRAMES"]

        self.e1 = encoder_block(1, 64, self.gated)
        self.e2 = encoder_block(64, 128, self.gated)
        self.e3 = encoder_block(128, 256, self.gated)
        self.e4 = encoder_block(256, 512, self.gated)

        self.g1 = encoder_block_1D(1, 64, self.gated)
        self.g2 = encoder_block_1D(64, 128, self.gated)
        self.g3 = encoder_block_1D(128, 256, self.gated)
        self.g4 = encoder_block_1D(256, 512, self.gated)

        self.sp_recuded = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        """""
        self.r1  = ResidualLayer_combined(in_channels=512,
                                            out_channels=1024,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            num_classes=self.num_classes)
        self.r2 = ResidualLayer_combined(in_channels=512,
                                                out_channels=1024,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                num_classes=self.num_classes)
        self.r3 = ResidualLayer_combined(in_channels=512,
                                                out_channels=1024,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                num_classes=self.num_classes)
        
        self.s1 = ResidualLayer_combined(in_channels=512,
                                                out_channels=1024,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                num_classes=self.num_classes)
        
        self.b_sp = conv_block_gated(512, 1024) if self.gated else conv_block(512, 1024)
        


        self.s2 = ResidualLayer_combined(in_channels=512,
                                                out_channels=1024,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                num_classes=self.num_classes)
        
        self.s3 = ResidualLayer_combined(in_channels=512,
                                                out_channels=1024,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                num_classes=self.num_classes)
        
        self.s4 = ResidualLayer_combined(in_channels=512,
                                                out_channels=1024,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                num_classes=self.num_classes)
        
        self.b_f0 = conv_block_gated(512, 1024) if self.gated else conv_block(512, 1024)
        """ ""

        self.r1 = ResidualLayer(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes + 512,
        )
        self.r2 = ResidualLayer(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes,
        )
        self.r3 = ResidualLayer(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes,
        )

        self.b_sp = (
            conv_block_gated(512 + self.num_classes, 1024)
            if self.gated
            else conv_block(512, 1024)
        )

        self.s1 = ResidualLayer(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes + 512,
        )

        self.s2 = ResidualLayer(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes,
        )

        self.s3 = ResidualLayer(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes,
        )

        self.s4 = ResidualLayer(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes,
        )

        self.b_f0 = (
            conv_block_gated(512 + self.num_classes, 1024)
            if self.gated
            else conv_block(512, 1024)
        )

        self.cc1 = ClassConditioning(
            height=self.num_mcep, width=self.n_frames, inp=self.num_classes
        )
        self.cc2 = ClassConditioning(
            height=self.num_mcep // 2, width=self.n_frames // 2, inp=self.num_classes
        )
        self.cc3 = ClassConditioning(
            height=self.num_mcep // 4, width=self.n_frames // 4, inp=self.num_classes
        )
        self.cc4 = ClassConditioning(
            height=self.num_mcep // 8, width=self.n_frames // 8, inp=self.num_classes
        )

        self.ccg1 = ClassConditioning(
            height=1, width=self.n_frames, inp=self.num_classes
        )
        self.ccg2 = ClassConditioning(
            height=1, width=self.n_frames // 2, inp=self.num_classes
        )
        self.ccg3 = ClassConditioning(
            height=1, width=self.n_frames // 4, inp=self.num_classes
        )
        self.ccg4 = ClassConditioning(
            height=1, width=self.n_frames // 8, inp=self.num_classes
        )

        self.d1 = decoder_block(
            1024, 512, gated=self.gated, num_classes=self.num_classes
        )
        self.d2 = decoder_block(
            512, 256, gated=self.gated, num_classes=self.num_classes
        )
        self.d3 = decoder_block(
            256, 128, gated=self.gated, num_classes=self.num_classes
        )
        self.d4 = decoder_block(128, 64, gated=self.gated, num_classes=self.num_classes)
        self.outputs_sp = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        self.h1 = decoder_block_1D(
            1024, 512, gated=self.gated, num_classes=self.num_classes
        )
        self.h2 = decoder_block_1D(
            512, 256, gated=self.gated, num_classes=self.num_classes
        )
        self.h3 = decoder_block_1D(
            256, 128, gated=self.gated, num_classes=self.num_classes
        )
        self.h4 = decoder_block_1D(
            128, 64, gated=self.gated, num_classes=self.num_classes
        )
        self.outputs_f0 = nn.Conv1d(64, 1, kernel_size=1, padding=0)

        self.cc_sp = ClassConditioning(
            height=2, width=self.n_frames // 16, inp=self.num_classes
        )
        self.cc_f0 = ClassConditioning(
            height=1, width=self.n_frames // 16, inp=self.num_classes
        )

        self.dropout = nn.Dropout(p=0.6)

    def forward(self, data, one_hot_labels=False):
        coded_sp = data[:, : self.num_mcep, :].unsqueeze(1)
        f0 = data[:, self.num_mcep :, :]  # f0 is the last component

        # feature extraction
        s1, p1 = self.e1(coded_sp)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        g1, q1 = self.g1(f0)
        g2, q2 = self.g2(q1)
        g3, q3 = self.g3(q2)
        g4, q4 = self.g4(q3)

        q4 = q4.unsqueeze(2)

        reduce_sp = self.sp_recuded(p4)
        repeat_f0 = q4.repeat(1, 1, p4.shape[2], 1)

        # print("repeat_f0.shape", repeat_f0.shape, "reduce_sp.shape", reduce_sp.shape)
        # print("p4.shape", p4.shape, "q4.shape", q4.shape)

        """""
        one_hot_labels_expanded = one_hot_labels.unsqueeze(-1).unsqueeze(-1)
        one_hot_labels_f0 = one_hot_labels_expanded.repeat(1, 1, repeat_f0.size(2), repeat_f0.size(3))
        one_hot_labels_sp = one_hot_labels_expanded.repeat(1, 1, reduce_sp.size(2), reduce_sp.size(3))

        repeat_f0 = torch.cat((repeat_f0, one_hot_labels_f0), dim=1)
        reduce_sp = torch.cat((reduce_sp, one_hot_labels_sp), dim=1)
        
        # residual layers
        r1 = self.r1(p4, repeat_f0)
        r2 = self.r1(r1, repeat_f0)
        r3 = self.r1(r2, repeat_f0)
        res_sp = self.b_sp(r3)

        p1 = self.s1(q4, reduce_sp)
        p2 = self.s2(p1, reduce_sp)
        p3 = self.s3(p2, reduce_sp)
        res_f0 = self.b_f0(p3).squeeze(2)
        """ ""

        ccr1 = self.cc_sp(one_hot_labels)

        p4 = torch.cat([p4, ccr1, repeat_f0], dim=1)
        # print("p4.shape", p4.shape)
        r1 = self.r1(p4)
        r1 = torch.cat([r1, ccr1], dim=1)
        r2 = self.r2(r1)
        r2 = torch.cat([r2, ccr1], dim=1)
        r3 = self.r3(r2)
        r3 = torch.cat([r3, ccr1], dim=1)
        res_sp = self.b_sp(r3)

        ccr2 = self.cc_f0(one_hot_labels)
        q4 = torch.cat([q4, ccr2, reduce_sp], dim=1)
        p1 = self.s1(q4)
        p1 = torch.cat([p1, ccr2], dim=1)
        p2 = self.s2(p1)
        p2 = torch.cat([p2, ccr2], dim=1)
        p3 = self.s3(p2)
        p3 = torch.cat([p3, ccr2], dim=1)
        res_f0 = self.b_f0(p3).squeeze(2)

        cc1 = self.cc1(one_hot_labels)
        cc2 = self.cc2(one_hot_labels)
        cc3 = self.cc3(one_hot_labels)
        cc4 = self.cc4(one_hot_labels)

        s1 = torch.cat([s1, cc1], dim=1)
        s2 = torch.cat([s2, cc2], dim=1)
        s3 = torch.cat([s3, cc3], dim=1)
        s4 = torch.cat([s4, cc4], dim=1)

        ccg1 = self.ccg1(one_hot_labels)
        ccg2 = self.ccg2(one_hot_labels)
        ccg3 = self.ccg3(one_hot_labels)
        ccg4 = self.ccg4(one_hot_labels)

        g1 = torch.cat([g1, ccg1.squeeze(2)], dim=1)
        g2 = torch.cat([g2, ccg2.squeeze(2)], dim=1)
        g3 = torch.cat([g3, ccg3.squeeze(2)], dim=1)
        g4 = torch.cat([g4, ccg4.squeeze(2)], dim=1)

        # decoder spectrogram
        d1 = self.d1(res_sp, self.dropout(s4))
        d2 = self.d2(d1, self.dropout(s3))
        d3 = self.d3(d2, self.dropout(s2))
        d4 = self.d4(d3, self.dropout(s1))
        decoded_sp = self.outputs_sp(d4).squeeze(1)

        # decoder f0
        h1 = self.h1(res_f0, self.dropout(g4))
        h2 = self.h2(h1, self.dropout(g3))
        h3 = self.h3(h2, self.dropout(g2))
        h4 = self.h4(h3, self.dropout(g1))
        decoded_f0 = self.outputs_f0(h4)

        output = torch.cat((decoded_sp, decoded_f0), dim=1)

        return output


"""""
        one_hot_expanded_2d = one_hot_labels.unsqueeze(-1).unsqueeze(-1)
        one_hot_expanded_1d = one_hot_labels.unsqueeze(-1)

        s1 = torch.cat([s1, one_hot_expanded_2d.repeat(1, 1, s1.size(2), s1.size(3))], dim=1)
        s2 = torch.cat([s2, one_hot_expanded_2d.repeat(1, 1, s2.size(2), s2.size(3))], dim=1)
        s3 = torch.cat([s3, one_hot_expanded_2d.repeat(1, 1, s3.size(2), s3.size(3))], dim=1)
        s4 = torch.cat([s4, one_hot_expanded_2d.repeat(1, 1, s4.size(2), s4.size(3))], dim=1)

        g1 = torch.cat([g1, one_hot_expanded_1d.repeat(1, 1, g1.size(2))], dim=1)
        g2 = torch.cat([g2, one_hot_expanded_1d.repeat(1, 1, g2.size(2))], dim=1)
        g3 = torch.cat([g3, one_hot_expanded_1d.repeat(1, 1, g3.size(2))], dim=1)
        g4 = torch.cat([g4, one_hot_expanded_1d.repeat(1, 1, g4.size(2))], dim=1)
""" ""


class SeperatePathwayGenerator(nn.Module):
    def __init__(self, csv):
        super().__init__()
        self.gated = csv["MODEL"]["GATED_CONV"]
        self.num_mcep = csv["INPUT"]["NUM_MCEP"]
        self.num_classes = len(csv["DATASET"]["EMOTIONS"])
        self.n_frames = csv["INPUT"]["N_FRAMES"]

        self.e1 = encoder_block(1, 64, self.gated)
        self.e2 = encoder_block(64, 128, self.gated)
        self.e3 = encoder_block(128, 256, self.gated)
        self.e4 = encoder_block(256, 512, self.gated)

        self.g1 = encoder_block_1D(1, 64, self.gated)
        self.g2 = encoder_block_1D(64, 128, self.gated)
        self.g3 = encoder_block_1D(128, 256, self.gated)
        self.g4 = encoder_block_1D(256, 512, self.gated)

        self.sp_recuded = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.r1 = ResidualLayer(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes,
        )
        self.r2 = ResidualLayer(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes,
        )
        self.r3 = ResidualLayer(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes,
        )

        self.b_sp = (
            conv_block_gated(512 + self.num_classes, 1024)
            if self.gated
            else conv_block(512, 1024)
        )

        self.s1 = ResidualLayer(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes,
        )

        self.s2 = ResidualLayer(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes,
        )

        self.s3 = ResidualLayer(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes,
        )

        self.s4 = ResidualLayer(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes,
        )

        self.b_f0 = (
            conv_block_gated(512 + self.num_classes, 1024)
            if self.gated
            else conv_block(512, 1024)
        )

        self.cc_sp = ClassConditioning(
            height=2, width=self.n_frames // 16, inp=self.num_classes
        )
        self.cc_f0 = ClassConditioning(
            height=1, width=self.n_frames // 16, inp=self.num_classes
        )

        self.cc1 = ClassConditioning(
            height=self.num_mcep, width=self.n_frames, inp=self.num_classes
        )
        self.cc2 = ClassConditioning(
            height=self.num_mcep // 2, width=self.n_frames // 2, inp=self.num_classes
        )
        self.cc3 = ClassConditioning(
            height=self.num_mcep // 4, width=self.n_frames // 4, inp=self.num_classes
        )
        self.cc4 = ClassConditioning(
            height=self.num_mcep // 8, width=self.n_frames // 8, inp=self.num_classes
        )
        self.cc5 = ClassConditioning(
            height=self.num_mcep // 16, width=self.n_frames // 16, inp=self.num_classes
        )

        self.ccg1 = ClassConditioning(
            height=1, width=self.n_frames, inp=self.num_classes
        )
        self.ccg2 = ClassConditioning(
            height=1, width=self.n_frames // 2, inp=self.num_classes
        )
        self.ccg3 = ClassConditioning(
            height=1, width=self.n_frames // 4, inp=self.num_classes
        )
        self.ccg4 = ClassConditioning(
            height=1, width=self.n_frames // 8, inp=self.num_classes
        )
        self.ccg5 = ClassConditioning(
            height=1, width=self.n_frames // 16, inp=self.num_classes
        )

        self.d11 = upSample(
            in_channels=1024,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes,
        )
        self.d22 = upSample(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes,
        )
        self.d33 = upSample(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes,
        )
        self.d44 = upSample(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes,
        )

        self.h11 = upSample1D(
            in_channels=1024,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=1,
            num_classes=self.num_classes,
        )
        self.h22 = upSample1D(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=1,
            num_classes=self.num_classes,
        )
        self.h33 = upSample1D(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=1,
            num_classes=self.num_classes,
        )
        self.h44 = upSample1D(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=1,
            num_classes=self.num_classes,
        )

        self.d1 = conv_block_gated(1024, 512, num_classes=self.num_classes)
        self.d2 = conv_block_gated(512, 256, num_classes=self.num_classes)
        self.d3 = conv_block_gated(256, 128, num_classes=self.num_classes)
        self.d4 = conv_block_gated(128, 64, num_classes=self.num_classes)
        self.outputs_sp = nn.Conv2d(64 + self.num_classes, 1, kernel_size=1, padding=0)

        self.h1 = conv_block_gated_1D(1024, 512, num_classes=self.num_classes)
        self.h2 = conv_block_gated_1D(512, 256, num_classes=self.num_classes)
        self.h3 = conv_block_gated_1D(256, 128, num_classes=self.num_classes)
        self.h4 = conv_block_gated_1D(128, 64, num_classes=self.num_classes)
        self.outputs_f0 = nn.Conv1d(64 + self.num_classes, 1, kernel_size=1, padding=0)

    def forward(self, data, one_hot_labels=False):
        coded_sp = data[:, : self.num_mcep, :].unsqueeze(1)
        f0 = data[:, self.num_mcep :, :]  # f0 is the last component

        # feature extraction
        s1, p1 = self.e1(coded_sp)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        g1, q1 = self.g1(f0)
        g2, q2 = self.g2(q1)
        g3, q3 = self.g3(q2)
        g4, q4 = self.g4(q3)

        q4 = q4.unsqueeze(2)

        # residual layers
        # print("p4.shape", p4.shape)
        ccr1 = self.cc_sp(one_hot_labels)

        p4 = torch.cat([p4, ccr1], dim=1)
        r1 = self.r1(p4)
        r1 = torch.cat([r1, ccr1], dim=1)
        r2 = self.r1(r1)
        r2 = torch.cat([r2, ccr1], dim=1)
        r3 = self.r1(r2)
        r3 = torch.cat([r3, ccr1], dim=1)
        res_sp = self.b_sp(r3)

        ccr2 = self.cc_f0(one_hot_labels)
        q4 = torch.cat([q4, ccr2], dim=1)
        p1 = self.s1(q4)
        p1 = torch.cat([p1, ccr2], dim=1)
        p2 = self.s2(p1)
        p2 = torch.cat([p2, ccr2], dim=1)
        p3 = self.s3(p2)
        p3 = torch.cat([p3, ccr2], dim=1)
        res_f0 = self.b_f0(p3).squeeze(2)

        cc1 = self.cc1(one_hot_labels)
        cc2 = self.cc2(one_hot_labels)
        cc3 = self.cc3(one_hot_labels)
        cc4 = self.cc4(one_hot_labels)
        cc5 = self.cc5(one_hot_labels)

        ccg1 = self.ccg1(one_hot_labels)
        ccg2 = self.ccg2(one_hot_labels)
        ccg3 = self.ccg3(one_hot_labels)
        ccg4 = self.ccg4(one_hot_labels)
        ccg5 = self.ccg5(one_hot_labels)

        # decoder spectrogram
        res_sp = torch.cat((res_sp, cc5), dim=1)
        d1 = self.d11(res_sp)
        d1 = torch.cat((d1, cc4), dim=1)
        d2 = self.d22(d1)
        d2 = torch.cat((d2, cc3), dim=1)
        d3 = self.d33(d2)
        d3 = torch.cat((d3, cc2), dim=1)
        d4 = self.d44(d3)
        d4 = torch.cat((d4, cc1), dim=1)
        decoded_sp = self.outputs_sp(d4).squeeze(1)

        # decoder f0
        res_f0 = torch.cat((res_f0, ccg5.squeeze(2)), dim=1)
        h1 = self.h11(res_f0)
        h1 = torch.cat((h1, ccg4.squeeze(2)), dim=1)
        h2 = self.h22(h1)
        h2 = torch.cat((h2, ccg3.squeeze(2)), dim=1)
        h3 = self.h33(h2)
        h3 = torch.cat((h3, ccg2.squeeze(2)), dim=1)
        h4 = self.h44(h3)
        h4 = torch.cat((h4, ccg1.squeeze(2)), dim=1)
        decoded_f0 = self.outputs_f0(h4)

        output = torch.cat((decoded_sp, decoded_f0), dim=1)

        return output


class CombinedTransformerGenerator(nn.Module):
    def __init__(self, csv):
        super().__init__()
        self.gated = csv["MODEL"]["GATED_CONV"]
        self.num_mcep = csv["INPUT"]["NUM_MCEP"]
        self.num_classes = len(csv["DATASET"]["EMOTIONS"])
        self.n_frames = csv["INPUT"]["N_FRAMES"]

        self.e1 = encoder_block(1, 64, self.gated)
        self.e2 = encoder_block(64, 128, self.gated)
        self.e3 = encoder_block(128, 256, self.gated)
        self.e4 = encoder_block(256, 512, self.gated)

        self.g1 = encoder_block_1D(1, 64, self.gated)
        self.g2 = encoder_block_1D(64, 128, self.gated)
        self.g3 = encoder_block_1D(128, 256, self.gated)
        self.g4 = encoder_block_1D(256, 512, self.gated)

        self.sp_recuded = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        hidden_size = 512  # first dim of input
        num_hidden_layers = 2
        num_attention_heads = 4
        intermediate_size = 1024  # 1024
        dropout = 0.1
        max_position_embeddings = 24  # last dim of input

        self.transformer_encoder = TransformerBlock(
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            dropout,
            max_position_embeddings,
        )

        self.b_sp = (
            conv_block_gated(512 + self.num_classes, 1024)
            if self.gated
            else conv_block(512, 1024)
        )

        self.b_f0 = (
            conv_block_gated(512 + self.num_classes, 1024)
            if self.gated
            else conv_block(512, 1024)
        )

        self.cc_tr = ClassConditioning(
            height=2, width=self.n_frames // 16, inp=self.num_classes, num_channels=4
        )
        self.cc_sp = ClassConditioning(
            height=2, width=self.n_frames // 16, inp=self.num_classes
        )
        self.cc_f0 = ClassConditioning(
            height=1, width=self.n_frames // 16, inp=self.num_classes
        )

        self.cc1 = ClassConditioning(
            height=self.num_mcep, width=self.n_frames, inp=self.num_classes
        )
        self.cc2 = ClassConditioning(
            height=self.num_mcep // 2, width=self.n_frames // 2, inp=self.num_classes
        )
        self.cc3 = ClassConditioning(
            height=self.num_mcep // 4, width=self.n_frames // 4, inp=self.num_classes
        )
        self.cc4 = ClassConditioning(
            height=self.num_mcep // 8, width=self.n_frames // 8, inp=self.num_classes
        )
        self.cc5 = ClassConditioning(
            height=self.num_mcep // 16, width=self.n_frames // 16, inp=self.num_classes
        )

        self.ccg1 = ClassConditioning(
            height=1, width=self.n_frames, inp=self.num_classes
        )
        self.ccg2 = ClassConditioning(
            height=1, width=self.n_frames // 2, inp=self.num_classes
        )
        self.ccg3 = ClassConditioning(
            height=1, width=self.n_frames // 4, inp=self.num_classes
        )
        self.ccg4 = ClassConditioning(
            height=1, width=self.n_frames // 8, inp=self.num_classes
        )
        self.ccg5 = ClassConditioning(
            height=1, width=self.n_frames // 16, inp=self.num_classes
        )

        self.d11 = upSample(
            in_channels=1024,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes,
        )
        self.d22 = upSample(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes,
        )
        self.d33 = upSample(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes,
        )
        self.d44 = upSample(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            num_classes=self.num_classes,
        )

        self.h11 = upSample1D(
            in_channels=1024,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=1,
            num_classes=self.num_classes,
        )
        self.h22 = upSample1D(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=1,
            num_classes=self.num_classes,
        )
        self.h33 = upSample1D(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=1,
            num_classes=self.num_classes,
        )
        self.h44 = upSample1D(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=1,
            num_classes=self.num_classes,
        )

        self.d1 = conv_block_gated(1024, 512, num_classes=self.num_classes)
        self.d2 = conv_block_gated(512, 256, num_classes=self.num_classes)
        self.d3 = conv_block_gated(256, 128, num_classes=self.num_classes)
        self.d4 = conv_block_gated(128, 64, num_classes=self.num_classes)
        self.outputs_sp = nn.Conv2d(64 + self.num_classes, 1, kernel_size=1, padding=0)

        self.h1 = conv_block_gated_1D(1024, 512, num_classes=self.num_classes)
        self.h2 = conv_block_gated_1D(512, 256, num_classes=self.num_classes)
        self.h3 = conv_block_gated_1D(256, 128, num_classes=self.num_classes)
        self.h4 = conv_block_gated_1D(128, 64, num_classes=self.num_classes)
        self.outputs_f0 = nn.Conv1d(64 + self.num_classes, 1, kernel_size=1, padding=0)

    def forward(self, data, one_hot_labels=False):
        coded_sp = data[:, : self.num_mcep, :].unsqueeze(1)
        f0 = data[:, self.num_mcep :, :]  # f0 is the last component

        # feature extraction
        s1, p1 = self.e1(coded_sp)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        g1, q1 = self.g1(f0)
        g2, q2 = self.g2(q1)
        g3, q3 = self.g3(q2)
        g4, q4 = self.g4(q3)

        q4 = q4.unsqueeze(2)

        cc_tr = self.cc_tr(one_hot_labels)

        padding_size = p4.size(2) - q4.size(2)  # Calculate the padding size
        original_q4_size = q4.size(2)
        q4_padded = F.pad(q4, (0, 0, 0, padding_size), "constant", 0)

        transformer_sp, transformer_f0 = self.transformer_encoder(p4, q4_padded, cc_tr)
        transformer_f0 = transformer_f0[:, :, :original_q4_size, :]

        cc_sp = self.cc_sp(one_hot_labels)
        cc_f0 = self.cc_f0(one_hot_labels)

        # transformer_sp = p4
        # transformer_f0 = q4
        transformer_sp = torch.cat((transformer_sp, cc_sp), dim=1)
        transformer_f0 = torch.cat((transformer_f0, cc_f0), dim=1)

        res_sp = self.b_sp(transformer_sp)
        res_f0 = self.b_f0(transformer_f0).squeeze(2)

        cc1 = self.cc1(one_hot_labels)
        cc2 = self.cc2(one_hot_labels)
        cc3 = self.cc3(one_hot_labels)
        cc4 = self.cc4(one_hot_labels)
        cc5 = self.cc5(one_hot_labels)

        ccg1 = self.ccg1(one_hot_labels)
        ccg2 = self.ccg2(one_hot_labels)
        ccg3 = self.ccg3(one_hot_labels)
        ccg4 = self.ccg4(one_hot_labels)
        ccg5 = self.ccg5(one_hot_labels)

        # decoder spectrogram
        res_sp = torch.cat((res_sp, cc5), dim=1)
        d1 = self.d11(res_sp)
        d1 = torch.cat((d1, cc4), dim=1)
        d2 = self.d22(d1)
        d2 = torch.cat((d2, cc3), dim=1)
        d3 = self.d33(d2)
        d3 = torch.cat((d3, cc2), dim=1)
        d4 = self.d44(d3)
        d4 = torch.cat((d4, cc1), dim=1)
        decoded_sp = self.outputs_sp(d4).squeeze(1)

        # decoder f0
        res_f0 = torch.cat((res_f0, ccg5.squeeze(2)), dim=1)
        h1 = self.h11(res_f0)
        h1 = torch.cat((h1, ccg4.squeeze(2)), dim=1)
        h2 = self.h22(h1)
        h2 = torch.cat((h2, ccg3.squeeze(2)), dim=1)
        h3 = self.h33(h2)
        h3 = torch.cat((h3, ccg2.squeeze(2)), dim=1)
        h4 = self.h44(h3)
        h4 = torch.cat((h4, ccg1.squeeze(2)), dim=1)
        decoded_f0 = self.outputs_f0(h4)

        output = torch.cat((decoded_sp, decoded_f0), dim=1)

        return output
