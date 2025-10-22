import torch
import torch.nn as nn
import torch.nn.functional as F


from source.layers.klayer import (
    KLayer,
)


from source.layers.common_layers import (
    RGBNormalize,
    ReadOutConv,
    Reshape,
)

from source.layers.common_fns import (
    positionalencoding2d,
)


class AKOrN(nn.Module):

    def __init__(
        self,
        n=4,
        ch=256,
        L=1,
        T=8,
        psize=4,
        gta=True,
        J="attn",
        ksize=1,
        c_norm="gn",
        gamma=1.0,
        imsize=128,
        use_omega=False,
        init_omg=1.0,
        global_omg=False,
        maxpool=True,
        project=True,
        heads=8,
        use_ro_x=False,
        learn_omg=True,
        no_ro=False,
        autorescale=True,
    ):
        super().__init__()
        # assuming input's range is [0, 1]
        self.patchfy = nn.Sequential(
            RGBNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            nn.Conv2d(3, ch, kernel_size=psize, stride=psize, padding=0),
        )

        if not gta:
            self.pos_enc = True
            self.pemb_x = nn.Parameter(
                positionalencoding2d(ch, imsize // psize, imsize // psize).reshape(
                    -1, imsize // psize, imsize // psize
                )
            )
            self.pemb_c = nn.Parameter(
                positionalencoding2d(ch, imsize // psize, imsize // psize).reshape(
                    -1, imsize // psize, imsize // psize
                )
            )
        else:
            self.pos_enc = False

        self.n = n
        self.ch = ch
        self.L = L
        if isinstance(T, int):
            self.T = [T] * L
        else:
            self.T = T
        if isinstance(J, str):
            self.J = [J] * L
        else:
            self.J = J
        self.gamma = torch.nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.psize = psize
        self.imsize = imsize

        self.layers = nn.ModuleList()
        feature_hw = imsize // psize

        feature_hws = [feature_hw] * self.L
        chs = [ch] * (self.L + 1)

        for l in range(self.L):
            ch = chs[l]
            if l == self.L - 1:
                ch_next = chs[l + 1]
            else:
                ch_next = chs[l + 1]

            klayer = KLayer(
                n=n,
                ch=ch,
                J=self.J[l],
                gta=gta,
                c_norm=c_norm,
                use_omega=use_omega,
                init_omg=init_omg,
                global_omg=global_omg,
                heads=heads,
                learn_omg=learn_omg,
                ksize=ksize,
                hw=[feature_hws[l], feature_hws[l]],
                apply_proj=project,
            )
            readout = (
                ReadOutConv(ch, ch_next, self.n, 1, 1, 0)
                if not no_ro
                else nn.Identity()
            )
            linear_x = (
                nn.Conv2d(ch, ch_next, 1, 1, 0)
                if use_ro_x and l < self.L - 1
                else nn.Identity()
            )
            self.layers.append(nn.ModuleList([klayer, readout, linear_x]))
        ch = ch_next

        if maxpool:
            pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            pool = nn.AdaptiveAvgPool2d((1, 1))

        self.out = nn.Sequential(
            nn.Identity(),
            pool,
            Reshape(-1, ch),
            nn.Linear(ch, 4 * ch),
            nn.ReLU(),
            nn.Linear(4 * ch, ch),
        )
        self.fixed_ptb = False
        self.autorescale = autorescale

    def feature(self, inp):
        if self.autorescale and (
            inp.shape[2] != self.imsize or inp.shape[3] != self.imsize
        ):
            inp = F.interpolate(
                inp,
                (self.imsize, self.imsize),
                mode="bilinear",
            )
        c = self.patchfy(inp)

        if self.fixed_ptb:
            g = torch.Generator(device="cpu").manual_seed(1234)
            x = torch.randn(*(c.shape), generator=g).to(c.device)
        else:
            x = torch.randn_like(c)

        if self.pos_enc:
            c = c + self.pemb_c[None]
            x = x + self.pemb_x[None]
        xs = [x]
        es = [torch.zeros(x.shape[0], device=x.device)]
        for l, (kblock, ro, lin_x) in enumerate(self.layers):
            _xs, _es = kblock(x, c, T=self.T[l], gamma=self.gamma)
            x = _xs[-1]
            c = ro(x)
            x = lin_x(x)
            xs.append(_xs)
            es.append(_es)

        return c, x, xs, es

    def forward(self, input, return_xs=False, return_es=False):
        c, x, xs, es = self.feature(input)
        c = self.out(c)

        ret = [c]
        if return_xs:
            ret.append(xs)
        if return_es:
            ret.append(es)

        if len(ret) == 1:
            return ret[0]
        return ret
