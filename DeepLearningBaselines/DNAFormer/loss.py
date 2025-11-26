from imports import *

class loss_function(nn.Module):  
    def __init__(self,config):
        super(loss_function,self).__init__()
        
        self.config = config

        if self.config.loss_type == 'ce_consistency':
            self.ce_loss = nn.CrossEntropyLoss()
        
            
    def forward(self, model_output, label):
        
        # Assumes model_output always contains pred_left/pred_right
        pred       = model_output['pred']
        pred_left  = model_output['pred_left']
        pred_right = model_output['pred_right']
        
        # Cross entropy + consistency
        if self.config.loss_type == 'ce_consistency':
            label_argmax = torch.argmax(label,dim=1).long()
            label = label.float()
            #print(f"Label is shape {label.shape} and label_argmax is {label_argmax.shape}")
            
            ce_loss  = self.config.ce_const_coeff_ce  * self.ce_loss(pred, label_argmax)
            # pred_left.softmax(dim=1) turns logits into probabilities, but ce_loss expects logits. CrossEntropyLoss will then  do a log_softmax again, so softmax twice (not wrong but can be unstable)
            # Also is label corret shape? 

            # Check predictions
            assert pred.dim() == 3,            f"pred must be [B, C, T], got {tuple(pred.shape)}"
            assert pred.dtype.is_floating_point, "pred must be floats (logits)"
            assert pred.size(1) == 4, (
                f"pred should have {4} classes, got {pred.size(1)}"
            )

            # Check hard targets
            assert label_argmax.dim() == 2,   f"label_argmax must be [B, T], got {tuple(label_argmax.shape)}"
            assert label_argmax.dtype == torch.long, "label_argmax must be torch.long"
            assert label_argmax.max() < pred.size(1), (
                f"class indices must be < {pred.size(1)}, got max index {label_argmax.max()}"
            )

            ### Question here: should it be label or label_argmax as input? 
            # Also do pred_left.softmax(dim=1), but CrossEntropyLoss expects raw logits
            # Also label is one hot encoded but Crossentropy loss expects tragets as class indices i.e. label_argmax, actually can also put in the one hot prob distribution 

            consistency_loss = self.config.ce_const_coeff_const * 0.5 * (self.ce_loss(pred_left.softmax(dim=1), label) + self.ce_loss(pred_right.softmax(dim=1), label))
            
            # Adjusted 
            #consistency_loss = self.config.ce_const_coeff_const * 0.5 * (
            #    self.ce_loss(pred_left, label_argmax) +
            #    self.ce_loss(pred_right, label_argmax)
            #)

            # Total loss
            loss = ce_loss + consistency_loss
            
        # Build logger
        if self.config.loss_type=='ce_consistency':
            logger = {'loss':loss, 'ce_loss':ce_loss, 'consistency_loss':consistency_loss}

        return logger