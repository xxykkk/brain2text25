from html import parser
from torch import nn
import torch
from torch.nn import functional as F

from vit import VisionTransformer
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import time

import os, argparse

def get_data_loader(batch_size, path):
    train_loader = DataLoader(
        ImageFolder(
            root=os.path.join(path, 'train'),
            transform=transforms.Compose([
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(448),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        ), 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        ImageFolder(
            root=os.path.join(path, 'val'),
            transform=transforms.Compose([
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomResizedCrop(224),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        ), 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    return train_loader, val_loader


def load_official_vit_weights(my_model: torch.nn.Module) -> torch.nn.Module:
    """
    将官方ViT模型的权重加载到你的自定义ViT模型中。
    
    Args:
        my_model (torch.nn.Module): 你实例化的自定义ViT模型。
        official_state_dict (dict): 从官方 .pth 文件加载的状态字典 (state_dict)。

    Returns:
        torch.nn.Module: 已经加载了权重的你的模型。
    """
    import re
    from collections import OrderedDict
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    
    official = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    official_state_dict = official.state_dict()

    # 获取你模型的 state_dict，用于检查 key 和 shape
    my_model_sd = my_model.state_dict()
    
    # 创建一个新的 state_dict，用于存放重命名后的权重
    new_sd = OrderedDict()
    
    loaded_keys = []
    unloaded_official_keys = []

    print("--- 🚀 开始权重映射 ---")

    for k_off, v_off in official_state_dict.items():
        # k_off = 官方模型的 key, v_off = 官方模型的 value (权重)
        new_k = k_off  # 默认新旧 key 相同
        
        # --- 1. 顶层参数 (class_token, pos_embed) ---
        # 
        # [!! 修正点 !!]
        # 映射: official 'class_token' -> my_model 'pos_embed.cls_token'
        if new_k == 'class_token':
            new_k = 'pos_embed.cls_token'
            
        # [!! 修正点 !!]
        # 映射: official 'encoder.pos_embedding' -> my_model 'pos_embed.pos_embed'
        # (根据你的报告，官方key是 'encoder.pos_embedding')
        elif new_k == 'encoder.pos_embedding':
            new_k = 'pos_embed.pos_embed'
            
        # --- 2. Patch Embedding ---
        # 'conv_proj.' -> 'patch_embed.proj.'
        elif new_k.startswith('conv_proj.'):
            new_k = new_k.replace('conv_proj.', 'patch_embed.proj.')
            
        # --- 3. Transformer Blocks ---
        # 'encoder.layers.encoder_layer_X.' -> 'blocks.X.'
        elif new_k.startswith('encoder.layers.'):
            # 使用正则表达式替换层编号
            # 例如: 'encoder.layers.encoder_layer_5.ln_1.weight'
            # -> 'blocks.5.ln_1.weight'
            new_k = re.sub(r'encoder\.layers\.encoder_layer_(\d+)\.', r'blocks.\1.', new_k)
            
            # 3a. LayerNorm 映射
            # '.ln_1.' -> '.norm1.'
            # '.ln_2.' -> '.norm2.'
            new_k = new_k.replace('.ln_1.', '.norm1.')
            new_k = new_k.replace('.ln_2.', '.norm2.')
            
            # 3b. MLP 映射
            # '.mlp.0.' -> '.mlp.fc1.' (第1个Linear)
            # '.mlp.3.' -> '.mlp.fc2.' (第2个Linear)
            new_k = new_k.replace('.mlp.0.', '.mlp.fc1.')
            new_k = new_k.replace('.mlp.3.', '.mlp.fc2.')
            
            # 3c. Attention 映射 (关键)
            # '.self_attention.in_proj_' -> '.attn.qkv.'
            # '.self_attention.out_proj.' -> '.attn.out.'
            new_k = new_k.replace('.self_attention.in_proj_', '.attn.qkv.')
            new_k = new_k.replace('.self_attention.out_proj.', '.attn.out.')

        # --- 4. 最终的 Encoder LayerNorm ---
        # 'encoder.ln.' -> 'norm.'
        elif new_k.startswith('encoder.ln.'):
            new_k = new_k.replace('encoder.ln.', 'norm.')
            
        # --- 5. 分类头 ---
        # 'heads.head.' -> 'head.'
        elif new_k.startswith('heads.head.'):
            new_k = new_k.replace('heads.head.', 'head.')
            
        # --- 检查新 key 是否在你的模型中 ---
        if new_k in my_model_sd:
            # 检查形状是否匹配
            if v_off.shape == my_model_sd[new_k].shape:
                new_sd[new_k] = v_off
                loaded_keys.append(new_k)
            else:
                print(f"⚠️ 形状不匹配! 跳过: {k_off} ({v_off.shape}) -> {new_k} ({my_model_sd[new_k].shape})")
                unloaded_official_keys.append(k_off)
        else:
            # 官方 key 经过重命名后，在你的模型中找不到
            # 如果 key 没变 (new_k == k_off)，也加入未加载列表
            unloaded_official_keys.append(k_off)

    # --- 加载权重并打印报告 ---
    
    # 使用 strict=False，因为我们知道分类头和 patch_embed.norm 不会
    load_result = my_model.load_state_dict(new_sd, strict=False)
    
    print("\n--- 📊 权重加载报告 (v2) ---")
    print(f"官方模型总参数: {len(official_state_dict)}")
    print(f"你的模型总参数: {len(my_model_sd)}")
    print(f"✅ 成功加载 {len(loaded_keys)} 个匹配的参数。")

    print("\n--- ❓ 你的模型中未被初始化的参数 (Missing Keys) ---")
    print(" (这些参数将保持随机初始化，这是预期行为)")
    for k in load_result.missing_keys:
        if 'head.' in k:
            print(f"  * {k} (新分类头 - 正常)")
        elif 'patch_embed.norm' in k:
            print(f"  * {k} (自定义的patch_embed norm - 正常)")
        elif '.num_batches_tracked' in k:
            pass # 忽略BN的buffer
        else:
            print(f"  * {k} (!!请检查这个key!!)") # 修正后，这里不应再出现 pos_embed

    print("\n--- ⚠️ 官方模型中未被加载的参数 (Unexpected Keys) ---")
    print(" (这些是官方模型有，但你的模型没有的参数)")
    # 我们需要从原始的 'unloaded_official_keys' 列表中移除那些
    # 已经被成功映射的key (即使它们的名字改变了)
    successfully_mapped_official_keys = {
        'class_token', 
        'encoder.pos_embedding'
    }
    
    final_unloaded = [k for k in set(unloaded_official_keys) if k not in successfully_mapped_official_keys]

    for k in final_unloaded:
        if 'heads.head' in k:
            print(f"  * {k} (旧分类头 - 正常)")
        else:
            print(f"  * {k} (!!请检查这个key!!)") # 修正后，这里不应再出现 class_token
            
    print("---------------------------------")
    
    return my_model



def main(args):
    model = VisionTransformer(224, 16, 3, 
                            num_classes=30, embed_dim=768, 
                            num_heads=12, depth=12, qkv_bias=True, 
                            attn_drop_rate=0.2, proj_drop_rate=0.2)
    train_loader, val_loader = get_data_loader(args.batch_size, args.ds_path)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)
    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    # 2️⃣ 余弦衰减：从当前LR衰减到 0
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    # 组合调度器
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine],
                            milestones=[args.warmup_epochs])
    load_official_vit_weights(model)

    model.cuda()
    
    '''
    train loop
    '''
    best_loss = torch.inf

    print("Start training...")

    for epoch in range(args.epochs):
        ss = time.time()
        model.train()
        
        train_total_loss = 0
        correct = 0
        total = 0

        for i, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()
            _, pred = out.max(1)

            # print(y)
            # print(pred)
            # input("NE")
            correct += pred.eq(y).sum().item()
            total += y.size(0)
        

        epoch_train_loss = train_total_loss / len(train_loader)
        epoch_train_acc = correct / total
        
        print(f'epoch: {epoch}, train loss: {epoch_train_loss:.4f}')
        print(f'epoch: {epoch}, train acc: {epoch_train_acc:.2%}')
        print(f"Epoch {epoch}, LR: {scheduler.get_last_lr()[0]:.6f}")
        scheduler.step()
        print(f"Epoch {epoch}, LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"Epoch {epoch}, time: {time.time() - ss:.2f}")


        model.eval()
        val_total_loss = 0
        correct = 0
        total = 0
        for i, (x, y) in enumerate(val_loader):
            x = x.cuda()
            y = y.cuda()
            out = model(x)
            loss = criterion(out, y)

            val_total_loss += loss.item()
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
        
        epoch_val_loss = val_total_loss / len(val_loader)
        epoch_val_acc = correct / total
        print(f'epoch: {epoch}, val loss: {epoch_val_loss:.4f}')
        print(f'epoch: {epoch}, val acc: {epoch_val_acc:.2%}')

        if os.path.exists(f'My_exp/vit_fruit30_epoch_{epoch-1}.pth'):
            os.remove(f'My_exp/vit_fruit30_epoch_{epoch-1}.pth')
        torch.save(model.state_dict(), f'My_exp/vit_fruit30_epoch_{epoch}.pth')
        if val_total_loss / len(val_loader) < best_loss:
            best_loss = val_total_loss / len(val_loader)
            torch.save(model.state_dict(), f'My_exp/vit_fruit30_best.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ds_path', type=str, default=r'/home/cook/data/Datasets/fruit30_split')
    # /home/liuke/data/other/fruit30_split
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    args = parser.parse_args()

    main(args)




    