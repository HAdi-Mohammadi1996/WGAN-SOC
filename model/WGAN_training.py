import torch

class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 bsize, latent_dim, gp_weight=10, critic_iterations=5, print_every=50,
                 use_cuda=False):
        self.bsize = bsize
        self.nz = latent_dim
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every

        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.G.to(self.device)
        self.D.to(self.device)


    def sample_generator(self):
        """Sample from the generator."""
        # Sample latent space
        noise = torch.randn(self.bsize, self.nz, 1, 1, 1, device=self.device)
        generated_data = self.G(noise)
        return generated_data
    

    def _critic_train_iteration(self, real_data):
        """Performs one training step for the critic."""
        
        # Generate fake data (no gradient needed for generator here)
        with torch.no_grad():
            generated_data = self.sample_generator()

        # Critic predictions
        d_real = self.D(real_data)
        d_generated = self.D(generated_data)

        # Compute gradient penalty
        gradient_penalty = self._gradient_penalty(real_data, generated_data)
        self.losses['GP'].append(gradient_penalty.item())

        # Critic loss: maximize D(real) - D(fake) → minimize -(D(real) - D(fake))
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty

        # Backpropagation and optimization
        self.D_opt.zero_grad()
        d_loss.backward(retain_graph=True)
        self.D_opt.step()

        # Record loss
        self.losses['D'].append(d_loss.item())


    def _generator_train_iteration(self, real_data):
        """Performs one training step for the generator."""
        self.G_opt.zero_grad()

        # Generate fake data
        generated_data = self.sample_generator()

        # Get critic's score on fake data
        d_generated = self.D(generated_data)
        g_loss = -d_generated.mean()

        # Backprop and update generator
        g_loss.backward()
        self.G_opt.step()

        # Log loss
        self.losses['G'].append(g_loss.item())


    def _gradient_penalty(self, real_data, generated_data):
        # Interpolate between real and generated samples
        alpha = torch.rand(self.bsize, 1, 1, 1, 1, device=self.device, requires_grad=True)
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        # interpolated = interpolated.to(self.device)

        # Critic output for interpolated data
        prob_interpolated = self.D(interpolated)

        # Compute gradients of critic output w.r.t. interpolated data
        grad_outputs = torch.ones_like(prob_interpolated)
        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Flatten each sample’s gradient and compute L2 norm
        gradients = gradients.view(self.bsize, -1)
        gradient_norm = gradients.norm(2, dim=1)

        # Save gradient norm for monitoring
        self.losses['gradient_norm'].append(gradient_norm.mean().item())

        # Compute and return gradient penalty
        gradient_penalty = self.gp_weight * ((gradient_norm - 1) ** 2).mean()
        return gradient_penalty


    def _train_epoch(self, data_loader):
        for i, real_data in enumerate(data_loader):
            self.num_steps += 1

            # Assume real_data comes as a tuple (data, label) → we use only data
            real_data = real_data[0].to(self.device)

            # Train critic
            self._critic_train_iteration(real_data)

            # Train generator after every N critic updates
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(real_data)

            # Print progress
            if i % self.print_every == 0:
                print(f"Iteration {i + 1}")
                print(f"D: {self.losses['D'][-1]:.4f}")
                print(f"GP: {self.losses['GP'][-1]:.4f}")
                print(f"Gradient norm: {self.losses['gradient_norm'][-1]:.4f}")
                if self.losses['G']:
                    print(f"G: {self.losses['G'][-1]:.4f}")

    def train(self, data_loader, epochs):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            self._train_epoch(data_loader)

