from helper import *
from model.compgcn_conv import CompGCNConv
from model.compgcn_conv_basis import CompGCNConvBasis

class BaseModel(torch.nn.Module):
	def __init__(self, params):
		super(BaseModel, self).__init__()

		self.p		= params
		self.act	= torch.tanh
		self.bceloss	= torch.nn.BCELoss()

	def loss(self, pred, true_label):
		return self.bceloss(pred, true_label)
		
class CompGCNBase(BaseModel):
	def __init__(self, edge_index, edge_type, num_rel, params=None, et_labels=None, dataset=None):
		super(CompGCNBase, self).__init__(params)

		self.edge_index		= edge_index
		self.edge_type		= edge_type
		self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
		self.init_embed		= get_param((self.p.num_ent,   self.p.init_dim))
		self.device		= self.edge_index.device
		self.attr_predictor = None
		self.attacker = None
		self.et_labels = et_labels.cuda()
		self.dataset = dataset
		print(self.dataset)
		if dataset == 'FB15k-237':
			self.attr_predictor = torch.nn.Sequential(
				torch.nn.Linear(self.p.embed_dim, 32),
				torch.nn.LeakyReLU(),
				torch.nn.Linear(32, 32),
				torch.nn.LeakyReLU(),
				torch.nn.Linear(32, 50))

			self.attacker = torch.nn.Sequential(
				torch.nn.Linear(self.p.embed_dim, 32),
				torch.nn.LeakyReLU(),
				torch.nn.Linear(32, 32),
				torch.nn.LeakyReLU(),
				torch.nn.Linear(32, 50))
		elif dataset == 'WN18RR':
			self.attr_predictor = torch.nn.Sequential(
				torch.nn.Linear(self.p.embed_dim, 32),
				torch.nn.LeakyReLU(),
				torch.nn.Linear(32, 32),
				torch.nn.LeakyReLU(),
				torch.nn.Linear(32, 20))

			self.attr2_predictor = torch.nn.Sequential(
				torch.nn.Linear(self.p.embed_dim, 32),
				torch.nn.LeakyReLU(),
				torch.nn.Linear(32, 32),
				torch.nn.LeakyReLU(),
				torch.nn.Linear(32, 4))

			self.attacker = torch.nn.Sequential(
				torch.nn.Linear(self.p.embed_dim, 32),
				torch.nn.LeakyReLU(),
				torch.nn.Linear(32, 32),
				torch.nn.LeakyReLU(),
				torch.nn.Linear(32, 20))

			self.attacker2 = torch.nn.Sequential(
				torch.nn.Linear(self.p.embed_dim, 32),
				torch.nn.LeakyReLU(),
				torch.nn.Linear(32, 32),
				torch.nn.LeakyReLU(),
				torch.nn.Linear(32, 4))


		self.reverse = GradientReversalLayer()
		self.reverse2 = GradientReversalLayer()

		if self.p.num_bases > 0:
			self.init_rel  = get_param((self.p.num_bases,   self.p.init_dim))
		else:
			if self.p.score_func == 'transe': 	self.init_rel = get_param((num_rel,   self.p.init_dim))
			else: 					self.init_rel = get_param((num_rel*2, self.p.init_dim))

		if self.p.num_bases > 0:
			self.conv1 = CompGCNConvBasis(self.p.init_dim, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None
		else:
			self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim,      num_rel, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

	def forward_attr(self, sub, rel, attr=0):
		if self.dataset == 'FB15k-237':
			truth = self.et_labels[sub].float()
			fn_attr = torch.nn.BCEWithLogitsLoss()
		elif self.dataset == 'WN18RR':
			truth = self.et_labels[sub,1]
			fn_attr = torch.nn.CrossEntropyLoss()

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)	

		sub_emb = self.reverse(sub_emb)
		out = self.attr_predictor(sub_emb)

		loss = fn_attr(out.float(), truth)
		return loss, out, truth

	def forward_attr2(self, sub, rel, attr=0):
		truth = self.et_labels[sub,0]
		fn_attr = torch.nn.CrossEntropyLoss()

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)	

		sub_emb = self.reverse(sub_emb)
		out = self.attr2_predictor(sub_emb)

		loss = fn_attr(out.float(), truth)
		return loss, out, truth

	def forward_attack(self, sub, rel, attr=0):
		if self.dataset == 'FB15k-237':
			truth = self.et_labels[sub].float()
			fn_attr = torch.nn.BCEWithLogitsLoss()
		elif self.dataset == 'WN18RR':
			truth = self.et_labels[sub,1]
			fn_attr = torch.nn.CrossEntropyLoss()

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)	

		out = self.attacker(sub_emb)

		loss = fn_attr(out.float(), truth)
		return loss, out, truth

	def forward_attack2(self, sub, rel, attr=0):
		truth = self.et_labels[sub,0]
		fn_attr = torch.nn.CrossEntropyLoss()

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)	

		out = self.attacker2(sub_emb)

		loss = fn_attr(out.float(), truth)
		return loss, out, truth

	def forward_base(self, sub, rel, drop1, drop2):

		r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
		x, r	= self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
		x	= drop1(x)
		x, r	= self.conv2(x, self.edge_index, self.edge_type, rel_embed=r) 	if self.p.gcn_layer == 2 else (x, r)
		x	= drop2(x) 							if self.p.gcn_layer == 2 else x

		sub_emb	= torch.index_select(x, 0, sub)
		rel_emb	= torch.index_select(r, 0, rel)

		return sub_emb, rel_emb, x


class CompGCN_TransE(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None, et_labels=None, dataset=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params,et_labels,dataset)
		self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
		self.feature_drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)	
		obj_emb				= sub_emb + rel_emb

		x	= self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)		
		score	= torch.sigmoid(x)

		return score

class CompGCN_DistMult(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None, et_labels=None, dataset=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params,et_labels,dataset)
		self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
		self.feature_drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel,  self.hidden_drop, self.feature_drop)	
		obj_emb				= sub_emb * rel_emb

		x = torch.mm(obj_emb, all_ent.transpose(1, 0))
		x += self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score

class CompGCN_ConvE(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None, et_labels=None, dataset=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params,et_labels,dataset)

		self.bn0		= torch.nn.BatchNorm2d(1)
		self.bn1		= torch.nn.BatchNorm2d(self.p.num_filt)
		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)
		
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
		self.m_conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)
		self.et_labels = et_labels.cuda()

		flat_sz_h		= int(2*self.p.k_w) - self.p.ker_sz + 1
		flat_sz_w		= self.p.k_h 	    - self.p.ker_sz + 1
		self.flat_sz		= flat_sz_h*flat_sz_w*self.p.num_filt
		self.fc			= torch.nn.Linear(self.flat_sz, self.p.embed_dim)

	def concat(self, e1_embed, rel_embed):
		e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
		rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
		return stack_inp

	def forward(self, sub, rel):
		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)


		stk_inp				= self.concat(sub_emb, rel_emb)
		x				= self.bn0(stk_inp)
		x				= self.m_conv1(x)
		x				= self.bn1(x)
		x				= F.relu(x)
		x				= self.feature_drop(x)
		x				= x.view(-1, self.flat_sz)
		x				= self.fc(x)
		x				= self.hidden_drop2(x)
		x				= self.bn2(x)
		x				= F.relu(x)

		x = torch.mm(x, all_ent.transpose(1,0))
		x += self.bias.expand_as(x)

		score = torch.sigmoid(x)

		return score
