import torch
import torch.nn as nn
import torch.nn.functional as F

class ObjectEncoder(nn.Module):
    """Encodes individual object features into a latent representation"""
    def __init__(self, input_dim=11, hidden_dim=128, latent_dim=64):
        super(ObjectEncoder, self).__init__()
        # Process numerical features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, x):
        # x shape: [batch_size, num_objects, feature_dim]
        batch_size, num_objects, feature_dim = x.shape
        
        # Reshape to process all objects at once
        x_flat = x.view(-1, feature_dim)
        encoded = self.encoder(x_flat)
        
        # Reshape back to separate objects
        encoded = encoded.view(batch_size, num_objects, -1)
        return encoded

class ResPredictor(nn.Module):
    """Predicts detector performance based on variable number of object detections"""
    def __init__(self, input_dim=11, hidden_dim=128, latent_dim=64, num_detectors=3):
        super(ResPredictor, self).__init__()
        
        # Encode each object
        self.object_encoder = ObjectEncoder(input_dim, hidden_dim, latent_dim)
        
        # Process aggregated features
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_detectors)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, mask=None):
        """
        Forward pass with attention-based pooling over objects
        
        Args:
            x: Tensor of shape [batch_size, num_objects, feature_dim]
            mask: Boolean mask of shape [batch_size, num_objects] indicating valid objects
                 (True for valid objects, False for padding)
                 
        Returns:
            Tensor of shape [batch_size, num_detectors] with predicted performance scores
        """
        batch_size, max_objects, feature_dim = x.shape
        
        # Create default mask if none provided (assuming all objects are valid)
        if mask is None:
            mask = torch.ones(batch_size, max_objects, dtype=torch.bool, device=x.device)
        
        # Encode each object
        encoded_objects = self.object_encoder(x)  # [batch_size, num_objects, latent_dim]
        
        # Apply mask to zero out padding
        mask_expanded = mask.unsqueeze(-1).expand_as(encoded_objects)
        encoded_objects = encoded_objects * mask_expanded.float()
        
        # Use attention pooling to aggregate object features
        attention_weights = self._compute_attention(encoded_objects, mask)
        
        # Apply attention weights
        weighted_objects = encoded_objects * attention_weights.unsqueeze(-1)
        pooled_features = weighted_objects.sum(dim=1)  # [batch_size, latent_dim]
        
        # Predict detector scores
        detector_scores = self.predictor(pooled_features)
        
        return detector_scores
    
    def _compute_attention(self, encoded_objects, mask):
        """Compute attention weights for objects"""
        # Simple attention mechanism (can be replaced with more sophisticated attention)
        scores = torch.sum(encoded_objects, dim=-1)  # [batch_size, num_objects]
        
        # Mask out padding objects with very negative values
        masked_scores = scores.masked_fill(~mask, -1e9)
        
        # Apply softmax to get weights
        weights = F.softmax(masked_scores, dim=1)  # [batch_size, num_objects]
        return weights

# Usage example with variable number of objects per batch
def collate_variable_objects(batch):
    """Custom collate function for handling variable number of objects"""
    batch_size = len(batch)
    
    # Find max number of objects in this batch
    max_objects = max([len(sample['objects']) for sample in batch])
    
    # Initialize tensors
    features = torch.zeros(batch_size, max_objects, 11)
    masks = torch.zeros(batch_size, max_objects, dtype=torch.bool)
    labels = torch.zeros(batch_size, 3)  # Assuming 3 detectors
    
    # Fill tensors
    for i, sample in enumerate(batch):
        num_objects = len(sample['objects'])
        features[i, :num_objects] = torch.tensor(sample['objects'])
        masks[i, :num_objects] = True
        labels[i] = torch.tensor(sample['detector_scores'])
    
    return {'features': features, 'masks': masks, 'labels': labels}

# Training loop example
def train_detector_predictor(model, train_loader, val_loader, num_epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            features = batch['features']
            masks = batch['masks']
            labels = batch['labels']
            
            # Forward pass
            predictions = model(features, masks)
            loss = criterion(predictions, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features']
                masks = batch['masks']
                labels = batch['labels']
                
                predictions = model(features, masks)
                loss = criterion(predictions, labels)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}")
