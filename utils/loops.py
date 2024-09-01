import torch
import wandb

from time import time


def train(model,train_dataloader,eval_dataloader,optimizer,lr_scheduler,args):

    model.train()
    batch = 0
    t0 = t_log = time()


    for epoch in range(args.epochs):

        val_loss, val_acc = evaluate(model,eval_dataloader,args)

        print("-"*40)
        print(f"Epoch {epoch}")
        print("-"*40)
        print(f"Current validation loss is: {val_loss:.2f}")
        print(f"Current validation accuracy is: {val_acc:.2f}")

        wandb.log({
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
                            
        for x, y in train_dataloader:

            batch += 1
        
            x, y = x.to(args.device), y.to(args.device)

            loss = model(input = x, label = y, return_type = "loss")
            
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # ------------------LOGGING--------------------   
        
            if batch % args.batch_per_log == 0: 
                t1 = time()
                dt = t1-t_log
                t_log = t1

                graphs_per_second = args.batch_size*args.batch_per_log/dt
                                    
                print(f"batch {batch:.0f} | loss: {loss.item():.2f} | dt: {dt*1000:.2f}ms | graphs/s: {graphs_per_second:.2f}")
                wandb.log({
                    "loss": loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "time": t1 - t0,
                    "batch": batch,
                    "Graphs per Second": graphs_per_second
                })
                    
            # ----------------------------------------------           

    train_time = int(time()) - int(t0)

    return model

def evaluate(model, loader, args):
    model_mode = model.training
    model.eval()

    loss_sum = 0
    ncorrect = 0
    nsamples = 0

    with torch.no_grad():
        for x, y in loader:

            x, y = x.to(args.device), y.to(args.device)

            logits, loss = model(input = x, label = y, return_type = "both")

            loss_sum += loss
            ncorrect += (logits.argmax(dim=1)==y).sum().item()
            nsamples += len(y)

    val_loss = loss_sum/len(loader)
    val_acc = ncorrect/nsamples

    if model_mode:
        model.train()

    return val_loss, val_acc