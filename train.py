from __future__ import print_function
from utils.utils import *
from tqdm import tqdm
from utils.showbs import *
from utils.helpers import *
from collections import OrderedDict
import json
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def init_io(root_path):
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    io = IOStream('%s/0run.log' % (root_path))
    return io

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def train(args):
    #1.*****************prepare**********************
    path_checkpoint = 'ck_train/%s/%s_%stk_%sbs%sw%ss%sq%s' % (
        args.dataset_train, args.backbone, args.task_num,args.batch_size, args.n_way, args.n_shot, args.n_query, args.exp)
    io = init_io(path_checkpoint)
    args_dict = vars(args)
    with open(os.path.join(path_checkpoint,'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)
        
    trainset = get_dataset(args.dataset_train, 'train', args.unsupervised, args, augment=args.augment)
    train_loader = get_train_dataloader(args, trainset)
    valset = get_dataset(args.dataset_test, 'test', False, args)
    val_loader= get_val_dataloader(args,valset)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    args.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Meta(args,trainset.__len__())
    max_epoch = 0
    max_acc = 0.0
    max_vap = 0.0
    con = 0
    if args.con != 'E0':
        path = os.path.join(path_checkpoint, "{}.pt".format(args.con))
        checkpoint = torch.load(path,map_location='cuda:0')
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k.replace('module.', '')   
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        max_acc = checkpoint['max_acc']
        max_vap = checkpoint['max_vap']
        con = max_epoch = checkpoint['epoch']
    model = model.to(device)

    if args.con != 'E0':
        model.model_optim.load_state_dict(checkpoint['model_optim_state_dict'])
        model.model_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(model.model_optim)
        print(model.model_scheduler.T_max)
        print(model.model_scheduler.eta_min)
        print(model.model_scheduler.last_epoch)
        print(model.model_scheduler.get_last_lr())#上传前注释掉


    if args.multi_gpu:
        model.encoder = nn.DataParallel(model.encoder, dim=0)
        para_model = model
    else:
        para_model = model

    io.cprint("-----------------------------------------------------------------------------")
    # 2.*****************train**********************
    va, vap =0,0
    for ep in range(con + 1, args.n_epochs):  
        trainset.change_assign_label(model.get_assign_labels().clone().cpu())
        train_loader = get_train_dataloader(args, trainset)
        pbar = tqdm(total=len(train_loader), desc='epoch %d' % ep)
        que_labels = torch.arange(args.n_way).view(-1, 1).expand(args.task_num, -1, args.n_query).reshape(-1).cuda()
        prob=args.pmax*(round(ep/args.n_epochs,3))
        for i,(sup_que_strong, _,neighboor, label_index) in enumerate(train_loader):
            model.train()
            if ep == (con + 1) and i==0:
                print(sup_que_strong.size())
                print(neighboor.size())
                print(que_labels[0:(args.n_way*args.n_query)])
                show(args.batch_size, args.n_shot+args.n_query, args.n_query,sup_que_strong.clone().detach(),neighboor.clone().detach())
            sup_que_strong = sup_que_strong.cuda()
            neighboor = neighboor.cuda()
            model_loss, model_accs= para_model.train_UML_with_center(sup_que_strong,neighboor,que_labels,label_index,args.task_num,
                                                                args.batch_size, args.n_way, args.n_shot, args.n_query,prob)
            pbar.set_description(f'epoch {ep}, model_loss: {model_loss.item():.4f}, model_acc: {model_accs:.4f}')
            pbar.update()
        pbar.close()
        model.model_scheduler.step()
        model.update_center()
        # 3.*****************eval**********************
        if ep % args.eval_interval == 0:
            model.eval()
            record = np.zeros(args.n_eval_episodes)
            query_labels = torch.arange(args.n_test_way).view(-1, 1).expand(-1, args.n_test_query).reshape(-1).cuda()
            with torch.no_grad():
                for i, batch in tqdm(enumerate(val_loader, 1), total=len(val_loader), desc='eval procedure'):
                    sup_que = batch[0].cuda()
                    accs = model.finetunning(args, sup_que, query_labels)
                    record[i - 1] = accs
            assert (i == record.shape[0])
            va, vap = compute_confidence_interval(record)
            io.cprint('epoch {},{} way {} shot, val,  acc={:.4f}+{:.4f}'.format(ep, args.n_test_way,
                                                                                       args.n_test_shot, va,
                                                                                       vap))
            if va >= max_acc:
                max_epoch = ep
                max_acc = va
                max_vap = vap
                path = os.path.join(path_checkpoint, "Best_{}_{:.4f}.pt".format(max_epoch,max_acc))
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_optim_state_dict": model.model_optim.state_dict(),
                    "scheduler_state_dict": model.model_scheduler.state_dict(),
                    "max_acc": max_acc,
                    "max_vap": max_vap,
                    "max_epoch": max_epoch
                }, path)
            io.cprint('best epoch {}, best val acc={:.4f} + {:.4f}'.format(max_epoch, max_acc, max_vap))
            io.cprint("-----------------------------------------------------------------------------")

        # 4.*****************save**********************
        if ep % 50 == 0 :
            path = os.path.join(path_checkpoint, "E{}.pt".format(int(ep)))
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_optim_state_dict": model.model_optim.state_dict(),
                "scheduler_state_dict": model.model_scheduler.state_dict(),
                "max_acc": va,
                "max_vap": vap,
                "epoch": ep
            }, path)

        path = os.path.join(path_checkpoint, "epoch-last.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_optim_state_dict": model.model_optim.state_dict(),
            "scheduler_state_dict": model.model_scheduler.state_dict(),
            "max_acc": va,
            "max_vap": vap,
            "epoch": ep
        }, path)


if __name__ == "__main__":
    args = parse_args()
    # 判断有没有gpu
    args.dataset_train='TieredImageNet' #['MiniImageNet', 'TieredImageNet', 'FC100', 'CIFAR-FS','CUB''ISIC','EuroSAT','chestx','CropDiseases']
    args.dataset_test='TieredImageNet'
    args.backbone="Res12"## conv4,Res12
    args.task_num=256
    args.batch_size=32#64
    args.pmax = 0.6
    args.n_epochs=400
    args.cluster_num = 512
    args.exp="_{:}_pmax{:}_cluster_num{:}".format(args.n_epochs,args.pmax,args.cluster_num)
    args.eval_interval=5
    args.con='epoch-last.pt'
    set_gpu(args.gpu)
    train(args)
