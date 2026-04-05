#!/usr/bin/env python3
"""
Push Medical Triage Environment to Hugging Face Spaces
"""
import os
import subprocess
import sys
from pathlib import Path
from huggingface_hub import HfApi, login, create_repo

def main():
    print("\n🚀 Medical Triage Environment - Hugging Face Push\n")
    
    # Get HF token from argument, env var, or user input
    token = None
    if len(sys.argv) > 1:
        token = sys.argv[1]
    elif "HF_TOKEN" in os.environ:
        token = os.environ["HF_TOKEN"]
    else:
        print("You need your Hugging Face token to proceed.")
        print("Get one at: https://huggingface.co/settings/tokens\n")
        token = input("Enter your Hugging Face token: ").strip()
    
    if not token:
        print("❌ Token is required!")
        sys.exit(1)
    
    # Authenticate
    print("\n🔐 Authenticating...")
    try:
        login(token=token)
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        sys.exit(1)
    
    # Get user info
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info["name"]
        print(f"✓ Authenticated as: @{username}")
    except Exception as e:
        print(f"❌ Failed to get user info: {e}")
        sys.exit(1)
    
    # Prepare repo details
    repo_name = "openev1"
    repo_id = f"{username}/{repo_name}"
    
    print(f"\n📦 Creating/updating repository: {repo_id}")
    print(f"   Type: Space (Docker)")
    
    try:
        # Create repo if it doesn't exist
        repo_url = create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True,
            token=token,
        )
        print(f"✓ Repository ready: {repo_url}")
    except Exception as e:
        print(f"❌ Failed to create/access repo: {e}")
        sys.exit(1)
    
    # Initialize git if needed and push
    print("\n📤 Pushing code to Hugging Face...")
    project_dir = Path(__file__).parent
    
    try:
        os.chdir(project_dir)
        
        # Add HF remote with token for authentication
        subprocess.run(
            ["git", "remote", "remove", "hf"],
            capture_output=True,
        )
        hf_url = f"https://x-access-token:{token}@huggingface.co/{repo_id}"
        subprocess.run(
            ["git", "remote", "add", "hf", hf_url],
            check=True,
            capture_output=True,
        )
        
        # Push to HF
        subprocess.run(
            ["git", "push", "hf", "main", "--force"],
            check=True,
        )
        
        print(f"✓ Successfully pushed to https://huggingface.co/{repo_id}")
        print(f"\n✨ Your Space is now available at:")
        print(f"   🔗 https://huggingface.co/spaces/{repo_id}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git push failed: {e}")
        print("\n💡 Try these steps manually:")
        print(f"   git remote add hf https://huggingface.co/{repo_id}")
        print(f"   git push hf main --force")
        sys.exit(1)

if __name__ == "__main__":
    main()
