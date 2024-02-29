import torch



class Hessian(torch.nn.Module):

	"""
	This class was implemented according to 
	Example:
		>>> hessian=Hessian(alpha=0.5, beta=0.5)
	
		>>> im=images[0:3]
		>>> im_torch=torch.from_numpy(im)
	
		>>> im_torch=im_torch.to(dtype=torch.float32)
	
		>>> im_torch=torch.permute(im_torch, (0,3,1,2))
		>>> print(im_torch.shape)
		>>> hessian_filtered=hessian(im_torch, mask=True)
	
		>>> plt.imshow(hessian_filtered[0])
	"""

	def __init__(self, alpha=0.5, beta=0.5, l=5, sig=10.):
		super(Hessian, self,).__init__()
		
		self.to_gray=torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32, requires_grad=False)
		self.r_a=torch.tensor([torch.inf], dtype=torch.float32, requires_grad=False)
		self.alpha=torch.tensor(alpha, dtype=torch.float32, requires_grad=False)
		self.beta=torch.tensor(beta, dtype=torch.float32, requires_grad=False)
		
		self.l=torch.tensor(l, requires_grad=False)
		self.sig=torch.tensor(sig, dtype=torch.float32, requires_grad=False)
		
		self.eps=torch.tensor(1e-10, dtype=torch.float32, requires_grad=False)
		
	def forward(self, x, mask=False):
		in_dims=x.shape
		
		#convert ot gray scale
		x=(torch.permute(x, (0,2,3,1))@self.to_gray).unsqueeze(1)

		if(mask):
			x=-x#if white ridges, if it is a mask then we compute for white ridges

		gaussian=self.gaussian_kernel()
		gaussian_filtered=torch.nn.functional.conv2d(x, gaussian.unsqueeze(0).unsqueeze(0), padding="same")
		hessian=torch.empty((in_dims[0],)+gaussian_filtered.shape[2:]+(gaussian.ndim, gaussian.ndim,), dtype=torch.float32)
			
		for b in range(in_dims[0]):
			grad=torch.gradient(gaussian_filtered[b,0], spacing=1.)
			for k, kgrad in enumerate(grad):
				kgrad_grad=torch.gradient(kgrad, spacing=1.)
				for l, klgrad in enumerate(kgrad_grad):
					hessian[b, :, :, k, l]=klgrad

		#now we have an hessian for each batch instance and also each hessian is symmetric
		eigvals=torch.linalg.eigvalsh(hessian, UPLO="U")
		eigvals=torch.take_along_dim(eigvals, torch.abs(eigvals).argsort(3), dim=3)

		lambda1=eigvals[...,0]
		lambda2=torch.maximum(eigvals[...,1:], self.eps)[...,0]
		
		r_b = torch.abs(lambda1) / lambda2  # eq. (15).
		s = torch.sqrt((eigvals ** 2).sum(3))  # eq. (12).)
		
		gamma = torch.amax(s, dim=(1,2), keepdims=True) / 2
		if torch.all(gamma==0.):
			gamma = 1.
		print(s.shape, torch.amax(s, dim=(1,2)), gamma.shape)
		vals = 1.0 - torch.exp(-self.r_a**2 / (2 * self.alpha**2))  # plate sensitivity
		vals = vals.unsqueeze(1) * torch.exp(-r_b**2 / (2 * self.beta**2))  # blobness
		vals *= 1.0 - torch.exp(-s**2 / (2 * gamma**2))  # structuredness
		filtered= torch.maximum(torch.zeros_like(vals), vals)

		filtered[filtered<=0]=1.

		return filtered

	def gaussian_kernel(self):
		"""\
		creates gaussian kernel with side length `l` and a sigma of `sig`
		"""
		
		ax=torch.linspace(-(self.l - 1) / 2., (self.l - 1) / 2., self.l)
		gauss = torch.exp(-0.5 * torch.square(ax) / torch.square(self.sig))
		kernel = torch.outer(gauss, gauss)
		return kernel / torch.sum(kernel)