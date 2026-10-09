import torch
import torch.nn.functional as F
from torch.autograd import Variable

class PolicyGradientRLHF:
    def __init__(self, generator_model, discriminator, style_classifier, optimizer, disc_optimizer=None, pad_token=0, sos_token=1):
        self.generator = generator_model
        self.discriminator = discriminator
        self.style_classifier = style_classifier
        self.optimizer = optimizer
        self.disc_optimizer = disc_optimizer
        self.pad_token = pad_token
        self.sos_token = sos_token
        
        self.baseline_mean = 0.0 
        self.baseline_std = 1.0
        self.entropy_coef = 0.01

    def sample_sequence(self, input_seqs, input_lengths, target_tones, target_rhetoric_vecs, max_len=50):
        batch_size = input_seqs.size(0)
        is_cuda = input_seqs.data.is_cuda
        
        mask = Variable((input_seqs.data == self.pad_token).byte())
        encoder_outputs, encoder_hidden = self.generator.encoder(input_seqs, input_lengths)
        
        decoder_hidden = encoder_hidden.contiguous().view(
            self.generator.encoder.n_layers, 2, batch_size, self.generator.encoder.hidden_size
        ).sum(1).expand(self.generator.decoder.n_layers, -1, -1).contiguous()
        
        decoder_input = Variable(torch.LongTensor([[self.sos_token] * batch_size])).transpose(0, 1)
        if is_cuda: decoder_input = decoder_input.cuda()

        step_log_probs, sampled_tokens, step_entropies = [], [], []
        
        for t in range(max_len):
            decoder_output, decoder_hidden, _ = self.generator.decoder(
                decoder_input, decoder_hidden, encoder_outputs, target_tones, target_rhetoric_vecs, mask
            )
            
            log_probs_step = F.log_softmax(decoder_output, dim=1)
            probs = torch.exp(log_probs_step)
            
            entropy = -(probs * log_probs_step).sum(1)
            step_entropies.append(entropy.unsqueeze(1))
            
            sampled = torch.multinomial(probs, 1) 
            sampled_tokens.append(sampled)
            
            chosen_log_probs = log_probs_step.gather(1, sampled)
            step_log_probs.append(chosen_log_probs)
            decoder_input = sampled

        seq_tensor = torch.cat(sampled_tokens, 1)
        log_prob_tensor = torch.cat(step_log_probs, 1)
        entropy_tensor = torch.cat(step_entropies, 1)
        return seq_tensor, log_prob_tensor, entropy_tensor

    def compute_reward(self, generated_seqs, target_tones, human_feedback_scores=None):
        gen_seqs_eval = Variable(generated_seqs.data, volatile=True) 
        human_likeness = self.discriminator(gen_seqs_eval, is_embedding=False).squeeze(1) 
        
        lengths = (gen_seqs_eval.data != self.pad_token).sum(1)
        lengths_var = Variable(torch.clamp(lengths, min=1))
        if generated_seqs.is_cuda:
            lengths_var = lengths_var.cuda()
        
        style_logits = self.style_classifier(gen_seqs_eval, lengths_var)
        target_tones_expanded = target_tones.unsqueeze(1)
        style_match_log_probs = style_logits.gather(1, target_tones_expanded).squeeze(1)
        style_match_score = torch.exp(style_match_log_probs)
        
        total_reward = (0.5 * human_likeness) + (0.5 * style_match_score)
        
        if human_feedback_scores is not None:
            total_reward = (0.3 * total_reward) + (0.7 * human_feedback_scores)
            
        return Variable(total_reward.data, requires_grad=False) 

    def train_step(self, input_seqs, input_lengths, target_tones, target_rhetoric_vecs, human_feedback=None):
        self.optimizer.zero_grad()
        batch_size = input_seqs.size(0)
        
        generated_seqs, log_probs, entropies = self.sample_sequence(input_seqs, input_lengths, target_tones, target_rhetoric_vecs)
        rewards = self.compute_reward(generated_seqs, target_tones, human_feedback)
        
        reward_mean = float(rewards.mean().data[0]) if rewards.mean().data.dim() > 0 else float(rewards.mean().data)
        reward_std = float(rewards.std().data[0]) if rewards.std().data.dim() > 0 else float(rewards.std().data)
        reward_std += 1e-8
        
        self.baseline_mean = 0.9 * self.baseline_mean + 0.1 * reward_mean
        self.baseline_std = 0.9 * self.baseline_std + 0.1 * reward_std
        
        advantages_data = (rewards.data - self.baseline_mean) / self.baseline_std
        advantages = Variable(advantages_data, requires_grad=False)
        
        mask_float = Variable((generated_seqs.data != self.pad_token).float())
        advantages_expanded = advantages.unsqueeze(1).expand_as(log_probs)
        
        pg_loss = -torch.sum(log_probs * advantages_expanded * mask_float) / batch_size
        entropy_bonus = -self.entropy_coef * torch.sum(entropies * mask_float) / batch_size
        
        total_loss = pg_loss + entropy_bonus
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm(self.generator.parameters(), 5.0)
        self.optimizer.step()
        
        pg_loss_val = float(pg_loss.data[0]) if pg_loss.data.dim() > 0 else float(pg_loss.data)
        entropy_val = float(entropy_bonus.data[0]) if entropy_bonus.data.dim() > 0 else float(entropy_bonus.data)
        
        return pg_loss_val, entropy_val, reward_mean

    def train_discriminator_step(self, real_seqs, input_seqs, input_lengths, target_tones, target_rhetoric_vecs, lambda_gp=10.0):
        if self.disc_optimizer is None:
            return 0.0
            
        self.disc_optimizer.zero_grad()
        batch_size = real_seqs.size(0)
        
        fake_seqs, _, _ = self.sample_sequence(input_seqs, input_lengths, target_tones, target_rhetoric_vecs)
        fake_seqs = Variable(fake_seqs.data)
        
        real_embeddings = self.discriminator.get_embeddings(real_seqs)
        fake_embeddings = self.discriminator.get_embeddings(fake_seqs)
        
        real_scores = self.discriminator(real_embeddings, is_embedding=True)
        fake_scores = self.discriminator(fake_embeddings, is_embedding=True)
        
        d_loss_pure = torch.mean(fake_scores) - torch.mean(real_scores)
        
        from evaluator import EvaluationMetrics
        gp = EvaluationMetrics.compute_gradient_penalty(self.discriminator, real_embeddings, fake_embeddings)
        
        total_d_loss = d_loss_pure + lambda_gp * gp
        total_d_loss.backward()
        
        self.disc_optimizer.step()
        
        d_loss_val = float(total_d_loss.data[0]) if total_d_loss.data.dim() > 0 else float(total_d_loss.data)
        return d_loss_val