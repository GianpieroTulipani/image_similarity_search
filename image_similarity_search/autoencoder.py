import dagshub
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from core.constants import SEED
from utils.loader import get_loader
from utils.seeding import init_seeds

dagshub.init(repo_owner="se4ai2425-uniba", repo_name="image_similarity_search", mlflow=True)


class Autoencoder(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, embedding_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 4096),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction


def train(model: nn.Module, train_path: str, test_path: str, save_path: str = "model.pth"):
    learning_rate = 0.001
    epochs = 20
    emb_dim = 512
    batch_size = 64
    name = "CNN Autoencoder"

    init_seeds(SEED)
    train_loader = get_loader(train_path, batch_size=batch_size)
    _ = get_loader(test_path, batch_size=batch_size)

    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("emb_dim", emb_dim)
        mlflow.log_param("lr", learning_rate)
        mlflow.log_param("name", name)
        mlflow.log_param("batch_size", batch_size)

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        model = Autoencoder(embedding_dim=emb_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            progress = tqdm.tqdm(enumerate(train_loader), desc="Progress", total=len(train_loader))
            for i, data in progress:
                data = data.to(device)
                embedding, reconstruction = model(data)
                loss = criterion(reconstruction, data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            mlflow.log_metric("accuracy", 42)

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--train_path", type=str, default="data/processed/train.csv")
    args.add_argument("--test_path", type=str, default="data/processed/test.csv")
    args.add_argument("--save_path", type=str, default="models/autoencoder.pth")
    args = args.parse_args()
    train(Autoencoder(), args.train_path, args.test_path, args.save_path)
