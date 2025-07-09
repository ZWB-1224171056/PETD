from model.backbone import *
from torch.optim.lr_scheduler import *
from model.Res10 import ResNet10
from model.split import *

class Meta(nn.Module):
    def __init__(self, args,data_len):
        super(Meta, self).__init__()
        if args.backbone == "conv4":
            self.encoder = ConvNet()
            if args.dataset_train in ['CIFAR-FS','FC100']:
                self.hdim=256
            else:
                self.hdim = 1600
            self.model_optim = torch.optim.Adam(self.encoder.parameters(), lr=0.002)
            print("conv4")
        elif args.backbone == "Res12":
            params = {}
            if args.dataset_train in ['CIFAR-FS', 'FC100']:
                params['drop_block'] = False
            self.encoder = ResNet(**params)
            self.hdim = 640
            self.model_optim = torch.optim.SGD(self.encoder.parameters(), lr=0.03,
                                               momentum=0.9, nesterov=True, weight_decay=0.0005)
            print("Res12")
        elif args.backbone == "Res10":
            self.encoder = ResNet10()
            self.hdim = 512
            self.model_optim = torch.optim.Adam(self.encoder.parameters())
            print("Res10")
        self.data_len=data_len
        self.args=args
        self.model_scheduler = CosineAnnealingLR(self.model_optim, args.n_epochs, eta_min=0)
        self.dual_lr = 32 if args.dataset_train == "TieredImageNet" else 4
        self.cluster_num = args.cluster_num #512
        self.ratio = args.ratio  
        self.lb = self.ratio / self.cluster_num  # y'/k
        centers = F.normalize(torch.randn(args.cluster_num, self.hdim), dim=1)
        self.register_buffer("pre_center", centers.clone())
        self.register_buffer("cur_center", centers.clone())
        self.register_buffer("dual", torch.zeros(args.cluster_num))
        self.register_buffer("counter", torch.zeros(args.cluster_num))
        self.register_buffer("assign_labels", torch.randint(0,args.cluster_num,(data_len,),dtype=torch.long))

    @torch.no_grad()
    def get_assign_labels(self):
        return self.assign_labels

    @torch.no_grad()
    def gen_label(self, ori_fea):
        return torch.argmax(torch.einsum("ij,kj->ik", ori_fea, self.cur_center) + self.dual, dim=1)

    @torch.no_grad()
    def get_label(self, targets):
        return self.assign_labels[targets]

    @torch.no_grad()
    def update_label(self, targets, labels):
        self.assign_labels[targets] = labels

    @torch.no_grad()
    def update_center(self):
        self.pre_center += self.cur_center.clone() - self.pre_center
        self.counter = torch.zeros(self.cluster_num).cuda()

    @torch.no_grad()
    def update_center_mini_batch(self, feats, labels):
        label_idx, label_count = torch.unique(labels, return_counts=True)
        self.dual[label_idx] -= self.dual_lr / len(labels) * label_count
        self.dual += self.dual_lr * self.lb
        if self.ratio < 1:
            self.dual[self.dual < 0] = 0
        alpha = self.counter[label_idx].float()
        self.counter[label_idx] += label_count
        alpha = alpha / self.counter[label_idx].float()
        self.cur_center[label_idx] = torch.einsum("ij,i->ij",self.cur_center[label_idx], alpha)
        self.cur_center.index_add_(0, labels, torch.einsum("ij,i->ij",feats.data , (1. / self.counter[labels])))
        self.cur_center[label_idx] = F.normalize(self.cur_center[label_idx], dim=1)

    def train_UML_with_center(self, sup_que,neighboor,que_labels,label_index,task_num, batch_size, n_way, n_shot, n_query,prob):
        fea_sq = self.encoder(sup_que)
        fea_neighboor = self.encoder(neighboor)
        fea_sq_neighboor = torch.cat((fea_sq,fea_neighboor),dim=0)
        pesudo_label=self.assign_labels[label_index]
        with torch.no_grad():
            fea_sq_normal = F.normalize(fea_sq, dim=-1)
            z_dim = fea_sq_normal.size(-1)
            aug_feat_mean = fea_sq_normal.view(batch_size, (n_shot + n_query), z_dim).mean(
                1)  # batch_size*dim
            pd_labels_now = self.gen_label(aug_feat_mean)
            self.update_label(label_index,pd_labels_now)
            self.update_center_mini_batch(aug_feat_mean, pd_labels_now)
            idx = torch.arange(self.data_len)
        self.model_optim.zero_grad()
        sup_idx, que_idx = split_neighboor(pesudo_label,task_num, batch_size, n_way, n_shot, n_query,prob)
        sup_fea = fea_sq_neighboor[sup_idx].view(task_num, n_way * n_shot, self.hdim)
        que_fea = fea_sq_neighboor[que_idx].view(task_num, n_way * n_query, self.hdim)
        proto = sup_fea.view(task_num, n_way, n_shot, self.hdim).mean(2)
        proto = F.normalize(proto, dim=-1)  # normalize for cosine similarity
        # que_fea = F.normalize(que_fea,dim=-1)
        num_proto = proto.shape[1]
        dists = torch.einsum("mij,mkj->mki", proto, que_fea).reshape(-1, num_proto)
        loss = F.cross_entropy(dists, que_labels)
        pred_q = torch.argmax(dists, dim=-1)
        correct = torch.eq(pred_q, que_labels).sum().item()
        accs = correct / (n_way * n_query * task_num)
        loss.backward()
        self.model_optim.step()

        return loss, accs

    def finetunning(self, args, sup_que, y_qry):
        z = self.encoder(sup_que)
        z_dim = z.size(-1)
        sup_idx, que_idx = split(args.n_test_way, args.n_test_shot, args.n_test_query)
        que_x = z[que_idx]
        z_proto = z[sup_idx].view(args.n_test_way, args.n_test_shot, z_dim).mean(1)
        z_proto = F.normalize(z_proto, dim=-1)
        # que_x = F.normalize(que_x, dim=-1)
        dists = torch.einsum("ij,kj->ki", z_proto, que_x)
        pred_q = F.softmax(dists, dim=1).argmax(dim=-1)
        correct = torch.eq(pred_q, y_qry).sum().item()
        accs = correct / (args.n_test_way * args.n_test_query)
        return accs

