import utils

import numpy as np

import torch
from torch import nn
from slot_attention.slot_attention import SlotAttentionAutoEncoder

class CSWM(nn.Module):
    """Main module for a Contrastively-trained Structured World Model (C-SWM).

    Args:
        embedding_dim: Dimensionality of abstract state space.
        hidden_dim: Number of hidden units in encoder and transition model.
        action_dim: Dimensionality of action space.
        num_objects: Number of object slots.
    """
    def __init__(self, embedding_dim, hidden_dim, action_dim, num_objects, hinge, sigma, 
                    ignore_action, copy_action, embodied, device):
        super(CSWM, self).__init__()

        self.hinge = hinge
        self.embodied = embodied
        self.transition_model = TransitionGNN(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_objects=num_objects,
            ignore_action=ignore_action,
            copy_action=copy_action,
            embodied=embodied,
            device=device)

        self.pos_loss = 0
        self.neg_loss = 0

        self.norm = 0.5 / (sigma**2)
        self.criterion = lambda pred, target : self.norm * (pred - target).pow(2).sum(2).mean(1)
    
    def energy(self, state, action, next_state, no_trans=False):
        """Energy function based on normalized squared L2 norm."""

        if no_trans: return self.criterion(state, next_state)
        return self.criterion(self.transition_model(state, action), next_state)

    def transition_loss(self, state, action, next_state):
        return self.energy(state, action, next_state).mean()

    def _contrastive_loss(self, state, action, next_state):
        # Sample negative state across episodes at random
        batch_size = state.size(0)
        perm = np.random.permutation(batch_size)
        neg_state = state[perm]
        self.pos_loss = self.energy(state, action, next_state)
        zeros = torch.zeros_like(self.pos_loss)
        
        self.pos_loss = self.pos_loss.mean()
        self.neg_loss = torch.max(
            zeros, self.hinge - self.energy(
                state, action, neg_state, no_trans=True)).mean()

        loss = self.pos_loss + self.neg_loss

        return loss

    def forward(self, obs):
        pass

    def save(self, iteration, opt, path):
        """
        saves a model to a given path
        :param path: (str) file path (.pt file)
        """
        data = {
            'iteration': iteration,
            'model_state_dict': self.state_dict(),
            'optim_state_dict': opt.state_dict(),
        }
        torch.save(data, path)


class ContrastiveSWM(CSWM):
    """Main module for a Contrastively-trained Structured World Model (C-SWM).

    Args: input_dims: Shape of input observation.
    """
    def __init__(self, embedding_dim, input_dims, hidden_dim, action_dim,
                 num_objects, hinge=1., sigma=0.5, encoder='large', num_feat=1,
                 ignore_action=False, copy_action=False, embodied=False, device="cpu"):
        super(ContrastiveSWM, self).__init__(embedding_dim, hidden_dim, action_dim,
                     num_objects, hinge, sigma, ignore_action, copy_action, embodied, device)
        
        num_channels = input_dims[0]
        width_height = input_dims[1:]
        self.name = "ContrastiveSWM"

        if encoder == 'small':
            self.obj_extractor = EncoderCNNSmall(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects,
                num_feat=num_feat)
            # CNN image size changes
            width_height = np.array(width_height)
            width_height = width_height // 10
        elif encoder == 'medium':
            self.obj_extractor = EncoderCNNMedium(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects,
                num_feat=num_feat)
            # CNN image size changes
            width_height = np.array(width_height)
            width_height = width_height // 5
        elif encoder == 'large':
            self.obj_extractor = EncoderCNNLarge(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects,
                num_feat=num_feat)

        self.obj_encoder = EncoderMLP(
            input_dim=np.prod(width_height)*4,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_objects=num_objects)

        self.width = width_height[0]
        self.height = width_height[1]

    def contrastive_loss(self, obs, action, next_obs):

        objs = self.obj_extractor(obs)
        next_objs = self.obj_extractor(next_obs)

        state = self.obj_encoder(objs)
        next_state = self.obj_encoder(next_objs)
        return self._contrastive_loss(state, action, next_state)

    def forward(self, obs):
        return self.obj_encoder(self.obj_extractor(obs))


class SlotSWM(CSWM):
    """ Builds CSWM replacing object extractor and encoder with Slot Attention.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
    """
    def __init__(self, args, hidden_dim, action_dim, sigma=0.5, init_weights=None,
                ignore_action=False, copy_action=False, embodied=False, device="cpu"):
        super(SlotSWM, self).__init__(args["slots_dim"], hidden_dim, action_dim, args["num_slots"] , 
            1, sigma, ignore_action, copy_action, embodied, device)
        
        self.obj_encoder = SlotAttentionAutoEncoder(
            args["resolution"], args["num_slots"], args["num_iterations"], args["slots_dim"],
            32 if args["small_arch"] else 64, args["small_arch"]).to(device)
            
        self.obj_encoder.encoder_cnn.encoder_pos.grid = self.obj_encoder.encoder_cnn.encoder_pos.grid.to(device)
        self.obj_encoder.decoder_cnn.decoder_pos.grid = self.obj_encoder.decoder_cnn.decoder_pos.grid.to(device)
        
        self.freeze_encoder = False

        if init_weights is not None:
            self.freeze_encoder = True
            self.obj_encoder.load_state_dict(torch.load(init_weights, map_location=device)['model_state_dict'])
    

    def _get_state_from_obs(self, obs):
        if self.freeze_encoder:
            return self.encode_no_grad(obs)
        slots = self.obj_encoder.encode(obs)
        return slots

    @torch.no_grad()
    def encode_no_grad(self, obs):
        # obs has shape [B, channels, width, height]
        self.obj_encoder.eval()
        # slots has shape [batch_size, num_slots, slot_size]
        slots = self.obj_encoder.encode(obs)
        return slots

    def contrastive_loss(self, obs, action, next_obs):
        state = self._get_state_from_obs(obs)
        next_state = self._get_state_from_obs(next_obs)
        if self.freeze_encoder:
            return self.transition_loss(state, action, next_state)
        return self._contrastive_loss(state, action, next_state)

    def forward(self, obs):
        return self._get_state_from_obs(obs)


class TransitionGNN(torch.nn.Module):
    """GNN-based transition function."""
    def __init__(self, input_dim, hidden_dim, action_dim, num_objects, ignore_action=False, 
                    copy_action=False, act_fn='relu', embodied=False, device="cpu"):
        super(TransitionGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.ignore_action = ignore_action
        self.copy_action = copy_action
        self.embodied = embodied
        self.device = device

        if self.ignore_action:
            self.action_dim = 0
        else:
            self.action_dim = action_dim

        edge_input_dim = input_dim*3

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim))

        node_input_dim = hidden_dim + input_dim + self.action_dim

        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, input_dim))

        self.edge_list = None
        self.batch_size = 0


    def _edge_model(self, source, target, edge_attr):
        #del edge_attr  # Unused.
        out = torch.cat([source, edge_attr, target], dim=1)
        return self.edge_mlp(out)


    def _node_model(self, node_attr, edge_index, edge_attr):
        if edge_attr is not None:
            row, col = edge_index
            agg = utils.unsorted_segment_sum(
                edge_attr, row, num_segments=node_attr.size(0))
            out = torch.cat([node_attr, agg], dim=1)
        else:
            out = node_attr
        return self.node_mlp(out)

    def _get_edge_list_fully_connected(self, batch_size, num_objects):
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.batch_size != batch_size:
            self.batch_size = batch_size

            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(num_objects, num_objects)

            # Remove diagonal.
            adj_full -= torch.eye(num_objects)
            self.edge_list = adj_full.nonzero()

            # Copy `batch_size` times and add offset.
            self.edge_list = self.edge_list.repeat(batch_size, 1)
            offset = torch.arange(
                0, batch_size * num_objects, num_objects).unsqueeze(-1)
            offset = offset.expand(batch_size, num_objects * (num_objects - 1))
            offset = offset.contiguous().view(-1)
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1)

            self.edge_list = self.edge_list.to(self.device)

        return self.edge_list

    def forward(self, states, action):

        batch_size = states.size(0)
        num_nodes = states.size(1)

        # states: [batch_size (B), num_objects, embedding_dim]
        # flat_states: Flatten states tensor to [B * num_objects, embedding_dim]
        flat_states = states.reshape(-1, self.input_dim)
        edge_attr = None

        # edge_index: [B * (num_objects*[num_objects-1]), 2] edge list
        edge_index = self._get_edge_list_fully_connected(batch_size, num_nodes - 1)
        row, col = edge_index

        if not self.ignore_action or self.embodied:

            if self.copy_action or self.embodied:
                action_vec = utils.to_one_hot(action, self.action_dim)
                action_vec = action_vec.repeat(1, self.num_objects).view(-1, self.action_dim)
            else:
                action_vec = utils.to_one_hot(action, self.action_dim * num_nodes)
                action_vec = action_vec.view(-1, self.action_dim)

        if not self.embodied:
            # Attach action to each state
            node_attr = torch.cat([flat_states, action_vec], dim=-1)
            edge_attr = self._edge_model(node_attr[row], node_attr[col], edge_attr)
            node_attr = self._node_model(node_attr, edge_index, edge_attr)
            flat_states = flat_states + node_attr

        else: # for descrete action spaces only
            # create empty transitions for the first round of message passing
            edge_attr = torch.zeros((flat_states.shape)).to(self.device)
            # compute relationships between objects (forces applied by an object to another)            
            edge_attr = self._edge_model(flat_states[row], flat_states[col], edge_attr[row])
            
            # get nodes transition (movement in space)
            node_attr = torch.cat([flat_states, action_vec], dim=-1)
            node_attr = self._node_model(node_attr, edge_index, edge_attr)
            
            # avoid applying actions repeadetly
            action_vec = torch.zeros(action_vec.shape).to(self.device)
            
            flat_states += node_attr
            
            for _ in range(num_nodes - 1):
                # update relationships based on previous transitions
                edge_attr = self._edge_model(flat_states[row], flat_states[col], node_attr[row])
                
                # get new transitions
                node_attr = self._node_model(torch.cat([flat_states, action_vec], dim=-1), 
                    edge_index, edge_attr)
                
                # update nodes with new transitions
                flat_states += node_attr
                            
        # [batch_size, num_nodes, embedding_dim]
        return flat_states.view(batch_size, num_nodes, -1)

class EncoderCNNSmall(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, num_feat, act_fn='sigmoid', act_fn_hid='relu'):
        super(EncoderCNNSmall, self).__init__()
        self.cnn1 = nn.Conv2d(input_dim, hidden_dim, (10, 10), stride=10)
        self.cnn2 = nn.Conv2d(hidden_dim, num_objects * num_feat, (1, 1), stride=1)
        self.ln1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = utils.get_act_fn(act_fn_hid)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        return self.act2(self.cnn2(h))
    
    
class EncoderCNNMedium(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, num_feat, act_fn='sigmoid', act_fn_hid='leaky_relu'):
        super(EncoderCNNMedium, self).__init__()

        self.num_objs = num_objects
        self.num_feat = num_feat
        self.cnn1 = nn.Conv2d(input_dim, hidden_dim, (9, 9), padding=4)
        self.act1 = utils.get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(hidden_dim, num_objects * num_feat, (5, 5), stride=5)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.cnn2(h))
        return h.reshape(h.shape[0], self.num_objs, self.num_feat * h.shape[-1], h.shape[-1])


class EncoderCNNLarge(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, num_feat, act_fn='sigmoid', act_fn_hid='relu'):
        super(EncoderCNNLarge, self).__init__()

        self.cnn1 = nn.Conv2d(input_dim, hidden_dim, (3, 3), padding=1)
        self.act1 = utils.get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act2 = utils.get_act_fn(act_fn_hid)
        self.ln2 = nn.BatchNorm2d(hidden_dim)

        self.cnn3 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act3 = utils.get_act_fn(act_fn_hid)
        self.ln3 = nn.BatchNorm2d(hidden_dim)

        self.cnn4 = nn.Conv2d(hidden_dim, num_feat * num_objects, (3, 3), padding=1)
        self.act4 = utils.get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.ln2(self.cnn2(h)))
        h = self.act3(self.ln3(self.cnn3(h)))
        return self.act4(self.cnn4(h))


class EncoderMLP(nn.Module):
    """MLP encoder, maps observation to latent state."""
    
    def __init__(self, input_dim, output_dim, hidden_dim, num_objects,act_fn='relu'):
        super(EncoderMLP, self).__init__()

        self.num_objects = num_objects
        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.ln = nn.LayerNorm(hidden_dim)

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        h_flat = ins.view(-1, self.num_objects, self.input_dim)
        h = self.act1(self.fc1(h_flat))
        h = self.act2(self.ln(self.fc2(h)))
        return self.fc3(h)


class DecoderMLP(nn.Module):
    """MLP decoder, maps latent state to image."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, output_size, act_fn='relu'):
        super(DecoderMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim + num_objects, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, np.prod(output_size))

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.output_size = output_size

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        obj_ids = torch.arange(self.num_objects)
        obj_ids = utils.to_one_hot(obj_ids, self.num_objects).unsqueeze(0)
        obj_ids = obj_ids.repeat((ins.size(0), 1, 1)).to(ins.get_device())

        h = torch.cat((ins, obj_ids), -1)
        h = self.act1(self.fc1(h))
        h = self.act2(self.fc2(h))
        h = self.fc3(h).sum(1)
        return h.view(-1, self.output_size[0], self.output_size[1], self.output_size[2])


class DecoderCNNSmall(nn.Module):
    """CNN decoder, maps latent state to image."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderCNNSmall, self).__init__()

        width, height = output_size[1] // 8, output_size[2] // 8

        output_dim = width * height

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        self.deconv1 = nn.ConvTranspose2d(num_objects, hidden_dim,
                                          kernel_size=1, stride=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, output_size[0],
                                          kernel_size=8, stride=8)

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.map_size = output_size[0], width, height

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)
        self.act3 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        h = self.act1(self.fc1(ins))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        h_conv = h.view(-1, self.num_objects, self.map_size[1],
                        self.map_size[2])
        h = self.act3(self.deconv1(h_conv))
        return self.deconv2(h)


class DecoderCNNMedium(nn.Module):
    """CNN decoder, maps latent state to image."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderCNNMedium, self).__init__()

        width, height = output_size[1] // 5, output_size[2] // 5

        output_dim = width * height

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        self.deconv1 = nn.ConvTranspose2d(num_objects, hidden_dim,
                                          kernel_size=5, stride=5)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, output_size[0],
                                          kernel_size=9, padding=4)

        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.map_size = output_size[0], width, height

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)
        self.act3 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        h = self.act1(self.fc1(ins))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        h_conv = h.view(-1, self.num_objects, self.map_size[1],
                        self.map_size[2])
        h = self.act3(self.ln1(self.deconv1(h_conv)))
        return self.deconv2(h)


class DecoderCNNLarge(nn.Module):
    """CNN decoder, maps latent state to image."""

    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderCNNLarge, self).__init__()

        width, height = output_size[1], output_size[2]

        output_dim = width * height

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        self.deconv1 = nn.ConvTranspose2d(num_objects, hidden_dim,
                                          kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                          kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                          kernel_size=3, padding=1)
        self.deconv4 = nn.ConvTranspose2d(hidden_dim, output_size[0],
                                          kernel_size=3, padding=1)

        self.ln1 = nn.BatchNorm2d(hidden_dim)
        self.ln2 = nn.BatchNorm2d(hidden_dim)
        self.ln3 = nn.BatchNorm2d(hidden_dim)

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.map_size = output_size[0], width, height

        self.act1 = utils.get_act_fn(act_fn)
        self.act2 = utils.get_act_fn(act_fn)
        self.act3 = utils.get_act_fn(act_fn)
        self.act4 = utils.get_act_fn(act_fn)
        self.act5 = utils.get_act_fn(act_fn)

    def forward(self, ins):
        h = self.act1(self.fc1(ins))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        h_conv = h.view(-1, self.num_objects, self.map_size[1],
                        self.map_size[2])
        h = self.act3(self.ln1(self.deconv1(h_conv)))
        h = self.act4(self.ln1(self.deconv2(h)))
        h = self.act5(self.ln1(self.deconv3(h)))
        return self.deconv4(h)