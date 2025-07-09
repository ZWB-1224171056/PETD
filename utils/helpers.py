from torch.utils.data import DataLoader
from model.meta import *
from dataloader import dataset_dict
from dataloader import CategoriesSampler, RandomSampler


def get_dataset(dataset, setname, unsupervised, args, augment='none'):
    Dataset = dataset_dict[dataset]
    return Dataset(setname, unsupervised, args, augment)

def examplar_collate(batch):
    X_strong, X_ori, X_neighboor, Y = [], [], [], []
    for b in batch:
        # X.append(torch.stack(b[0]))
        X_strong.extend(b[0])
        X_ori.extend(b[1])
        X_neighboor.extend(b[2])
        Y.append(b[3])
    img_strong = torch.stack(X_strong, dim=0)
    img_ori = torch.stack(X_ori, dim=0)
    img_neighbor = torch.stack(X_neighboor, dim=0)
    label_index = torch.LongTensor(Y)
    # img = torch.cat(tuple(X.permute(1, 0, 2, 3, 4)), dim=0)
    # (repeat * class , *dim)
    return img_strong, img_ori, img_neighbor, label_index

def get_train_dataloader(args, trainset):
    num_device = torch.cuda.device_count()
    num_workers = args.num_workers * num_device
    if args.unsupervised:
        train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=num_workers,
                                  collate_fn=examplar_collate,
                                  pin_memory=True, drop_last=True)
    else:
        train_sampler = CategoriesSampler(trainset.label,
                                          1,
                                          args.n_way,
                                          args.n_shot + args.n_query)

        train_loader = DataLoader(dataset=trainset,
                                  num_workers=num_workers,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)
    return train_loader

def get_test_dataloader(args):
    testsets = get_dataset(args.dataset_test, 'test', False, args)
    test_sampler = CategoriesSampler(testsets.label,
                                     args.n_test_episodes,
                                     args.n_test_way, args.n_test_shot + args.n_test_query)
    test_loader = DataLoader(dataset=testsets,
                             batch_sampler=test_sampler,
                             num_workers=args.num_workers,
                             pin_memory=True)
    return test_loader, testsets

def get_val_dataloader(args, valset):
    val_sampler = CategoriesSampler(valset.label,
                                    args.n_eval_episodes,
                                    args.n_test_way, args.n_test_shot + args.n_test_query)
    val_loader = DataLoader(dataset=valset,
                            batch_sampler=val_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)
    return val_loader


if __name__ == "__main__":
    split_pro(1, 64, 64, 1, 5, 0.5)
