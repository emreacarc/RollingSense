# Deployment Guide for RollingSense

This guide provides step-by-step instructions for deploying RollingSense to GitHub and Streamlit Cloud.

## Prerequisites

- GitHub account
- Streamlit Cloud account (free tier available)
- Git installed on your local machine
- Python 3.8+ installed locally (for initial setup)

## Step 1: Prepare Your Repository

### 1.1 Initialize Git (if not already done)

```bash
git init
git add .
git commit -m "Initial commit: RollingSense Predictive Maintenance System"
```

### 1.2 Create GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Name it `RollingSense` (or your preferred name)
5. Choose visibility (Public or Private)
6. **Do NOT** initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

### 1.3 Push to GitHub

```bash
git remote add origin https://github.com/emreacarc/RollingSense.git
git branch -M main
git push -u origin main
```

## Step 2: Prepare Models for Deployment

**Important:** Model files (`*.pkl`) and data files (`*.csv`) are excluded from Git via `.gitignore` to keep the repository size manageable. You have two options:

### Option A: Commit Models (Recommended for Small Models)

If your model files are under 100MB total:

1. **Temporarily modify `.gitignore`:**
   ```bash
   # Comment out or remove these lines:
   # models/*.pkl
   # models/*.csv
   ```

2. **Add and commit model files:**
   ```bash
   git add models/*.pkl models/*.csv models/*.json
   git commit -m "Add trained models and data files"
   git push
   ```

3. **Restore `.gitignore`:**
   ```bash
   # Uncomment the lines you removed
   git add .gitignore
   git commit -m "Restore .gitignore"
   git push
   ```

### Option B: Use External Storage (For Large Models)

1. Upload models to cloud storage (Google Drive, Dropbox, AWS S3, etc.)
2. Create a download script that fetches models at runtime
3. Use Streamlit Secrets to store credentials

## Step 3: Deploy to Streamlit Cloud

### 3.1 Sign Up for Streamlit Cloud

1. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. Click "Get started" or "Sign in"
3. Sign in with your GitHub account
4. Authorize Streamlit Cloud to access your GitHub repositories

### 3.2 Deploy Your App

1. Click "New app" button
2. Fill in the deployment form:
   - **Repository**: Select `emreacarc/RollingSense`
   - **Branch**: `main` (or your default branch)
   - **Main file**: `app.py`
   - **App URL**: (optional) Custom subdomain
3. Click "Deploy"

### 3.3 Wait for Deployment

Streamlit Cloud will:
- Clone your repository
- Install dependencies from `requirements.txt`
- Run `streamlit run app.py`
- Provide you with a public URL (e.g., `https://your-app.streamlit.app`)

### 3.4 Verify Deployment

1. Check the deployment logs for any errors
2. Visit your app URL
3. Test all modules:
   - Live Monitor
   - Sample Failure Scenarios
   - Failure Insights & Analytics
   - About Project

## Step 4: Handle Missing Models (If Using Option B)

If you didn't commit models and the app shows "Model not found" errors:

### Option 1: Train Models on Streamlit Cloud

Create a setup script that trains models if they don't exist:

1. Create `setup_models.py`:
   ```python
   import os
   from pathlib import Path
   from train import main as train_models
   
   if __name__ == "__main__":
       models_dir = Path("models")
       if not (models_dir / "best_model.pkl").exists():
           print("Models not found. Training models...")
           train_models()
       else:
           print("Models already exist.")
   ```

2. Add a command to run this before the app starts (Streamlit Cloud doesn't support this directly, but you can add it to your app initialization)

### Option 2: Download Models at Runtime

Modify `src/app_utils.py` to download models from external storage if they don't exist locally.

## Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution:** Ensure all dependencies are listed in `requirements.txt` with correct versions.

### Issue: "Model file not found"

**Solution:** 
- Check that model files exist in the `models/` directory
- If using `.gitignore`, ensure models are committed or available via external storage
- Verify file paths in `config.py`

### Issue: "App crashes on startup"

**Solution:**
- Check Streamlit Cloud logs for error messages
- Ensure `app.py` is the correct entry point
- Verify all imports are correct
- Check that required directories exist (they should be created by `config.py`)

### Issue: "Slow loading times"

**Solution:**
- Model files might be large; consider using lighter models or model compression
- Optimize data loading in `app_utils.py`
- Use caching with `@st.cache_data` or `@st.cache_resource`

## Updating Your Deployment

After making changes to your code:

1. **Commit changes:**
   ```bash
   git add .
   git commit -m "Description of changes"
   git push
   ```

2. **Streamlit Cloud will automatically redeploy** your app (usually within 1-2 minutes)

3. Check the deployment status in the Streamlit Cloud dashboard

## Environment Variables (Optional)

If you need to use environment variables or secrets:

1. In Streamlit Cloud dashboard, go to your app settings
2. Click "Secrets"
3. Add key-value pairs (e.g., API keys, database URLs)
4. Access them in your code with `st.secrets["key"]`

## Best Practices

1. **Keep repository size manageable**: Use `.gitignore` to exclude large files
2. **Version your models**: Consider using Git LFS for large model files
3. **Monitor usage**: Streamlit Cloud free tier has usage limits
4. **Test locally first**: Always test changes locally before pushing
5. **Use caching**: Leverage Streamlit's caching for expensive operations
6. **Error handling**: Add proper error handling for missing files/models

## Support

- Streamlit Cloud Documentation: [https://docs.streamlit.io/streamlit-cloud](https://docs.streamlit.io/streamlit-cloud)
- Streamlit Community: [https://discuss.streamlit.io](https://discuss.streamlit.io)
- GitHub Issues: Create an issue in your repository for bugs or feature requests
