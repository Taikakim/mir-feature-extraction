# GitHub Setup Guide

## Handling External Dependencies

This project includes a modified external repository (`timbral_models`) that has been patched for librosa 0.11.0 compatibility. Here's how to handle this for GitHub:

### Strategy: Use Setup Scripts

**We do NOT track the external repositories in git.** Instead, we:
1. Exclude `repos/` directory in `.gitignore`
2. Provide setup scripts that clone and patch external dependencies
3. Document the setup process in README.md

This approach:
- ✅ Keeps git history clean and small
- ✅ Respects external repository licenses
- ✅ Makes patches transparent and reproducible
- ✅ Allows users to easily update external deps

---

## Initial Push to GitHub

### 1. Create Repository on GitHub

1. Go to https://github.com/Taikakim
2. Click "New repository"
3. Repository name: `mir-feature-extraction` (or your preferred name)
4. Description: "Music Information Retrieval feature extraction for Stable Audio Tools"
5. **Keep it Private** initially (you can make it public later)
6. Do NOT initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

### 2. Push to GitHub

```bash
# Add GitHub remote (replace with your actual repo name if different)
git remote add origin https://github.com/Taikakim/mir-feature-extraction.git

# Push to GitHub
git push -u origin main
```

### 3. Verify

Visit https://github.com/Taikakim/mir-feature-extraction to see your repository.

---

## What Gets Pushed vs What's Excluded

### ✅ Tracked (Pushed to GitHub):

- All source code (`src/`)
- Setup scripts (`scripts/`)
- Documentation (`.md` files)
- Planning documents (`plans/`)
- Requirements file
- Label files (genre, instrument, mood)
- `.gitignore`

### ❌ Excluded (NOT Pushed):

- External repositories (`repos/`)
- Virtual environment (`mir/`)
- Audio files and test data (`test_data/`, `*.mp3`, `*.wav`, etc.)
- Separated stems
- Models directory
- Feature output files (`.INFO`, `.BEATS_GRID` - optional)
- Temporary files
- IDE settings

---

## For New Users Cloning Your Repository

After someone clones your repo, they need to:

### 1. Clone Your Repo

```bash
git clone https://github.com/Taikakim/mir-feature-extraction.git
cd mir-feature-extraction
```

### 2. Create Virtual Environment

```bash
python3 -m venv mir
source mir/bin/activate  # Windows: mir\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup External Repositories

```bash
# Method 1: Bash script
bash scripts/setup_external_repos.sh

# Method 2: Python script
python scripts/apply_patches.py
```

This clones `timbral_models` and applies the librosa 0.11.0 patches automatically.

---

## Updating External Patches

If you need to modify the patches in the future:

1. Edit the patch scripts:
   - `scripts/setup_external_repos.sh`
   - `scripts/apply_patches.py`

2. Update documentation:
   - `EXTERNAL_PATCHES.md`

3. Test the patches:
```bash
# Remove external repos
rm -rf repos/repos/timbral_models

# Re-run setup
bash scripts/setup_external_repos.sh

# Verify patches work
python src/timbral/audio_commons.py test_data/ --batch
```

4. Commit and push:
```bash
git add scripts/setup_external_repos.sh scripts/apply_patches.py EXTERNAL_PATCHES.md
git commit -m "Update external repository patches"
git push
```

---

## Alternative: Git Submodules (Not Recommended)

You could use git submodules for external repos, but it has drawbacks:

**Pros:**
- Tracks exact commit of external repo
- Standard git feature

**Cons:**
- Cannot commit patches to external repo
- More complex for users to manage
- Harder to maintain custom patches
- Submodule updates require careful coordination

**Our approach (setup scripts) is simpler and more maintainable for patched dependencies.**

---

## Repository Structure on GitHub

```
Taikakim/mir-feature-extraction/
├── .gitignore
├── README.md
├── USER_MANUAL.md
├── FEATURES_STATUS.md
├── EXTERNAL_PATCHES.md
├── GITHUB_SETUP.md (this file)
├── project.log (tracked exception)
├── IMPLEMENTATION_PLAN.md
├── requirements.txt
├── src/                    # All source code
├── scripts/                # Setup and utility scripts
├── plans/                  # Implementation plans
├── genre_labels.txt
├── instrument_labels.txt
└── mood_labels.txt

NOT tracked:
├── repos/                  # Cloned by setup scripts
├── mir/                    # Virtual environment
├── test_data/              # User's audio files
├── models/                 # Downloaded models
└── backup/                 # Local backups
```

---

## Making Repository Public

If you want to make the repository public later:

1. Go to repository Settings
2. Scroll to "Danger Zone"
3. Click "Change visibility"
4. Select "Make public"
5. Confirm

**Before making public, consider:**
- Remove any personal information or paths
- Ensure no API keys or credentials in code
- Review all committed files
- Choose an appropriate license (MIT, GPL, Apache, etc.)

---

## License Considerations

**Your Code:** You own the code you wrote. Choose a license:
- MIT License (permissive, widely used)
- GPL v3 (copyleft, requires derivatives to be open source)
- Apache 2.0 (permissive with patent grant)

**External Dependencies:**
- `timbral_models` (Audio Commons): Apache 2.0 License
- `demucs` (Meta): MIT License
- `essentia`: Affero GPL v3 (copyleft)
- `librosa`: ISC License (permissive)

**Recommended:** MIT or Apache 2.0 for maximum compatibility.

To add a license:
1. Create `LICENSE` file in repository root
2. Copy license text from https://choosealicense.com/
3. Commit and push

---

## Backup Strategy

**GitHub as Primary Backup:**
Your GitHub repository serves as your backup. Current setup:
- All code tracked
- All documentation tracked
- External deps reproducible via scripts

**Additional Backups:**
Consider also backing up locally:
- `.INFO` feature files (if you want to preserve extracted features)
- Models directory (essentia models, ~250MB)
- Test data (if you have rights to redistribute)

---

## Future Updates

### Regular Commits

```bash
# Make changes
git add <changed-files>
git commit -m "Description of changes"
git push
```

### Feature Branches

For major features:
```bash
# Create feature branch
git checkout -b feature/smart-cropping

# Work on feature...
git add <files>
git commit -m "Implement smart cropping"

# Push feature branch
git push -u origin feature/smart-cropping

# Create pull request on GitHub (optional, for review)
# Merge when ready
```

### Tags for Versions

```bash
# Tag a release
git tag -a v1.0 -m "Version 1.0 - Core feature extraction complete"
git push --tags
```

---

## Troubleshooting

### Push Failed: Authentication

If push fails with authentication error:

**Option 1: HTTPS with Personal Access Token**
1. Go to GitHub Settings > Developer settings > Personal access tokens
2. Generate new token with "repo" scope
3. Use token as password when pushing

**Option 2: SSH**
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "kim.ake@gmail.com"

# Add to GitHub (Settings > SSH and GPG keys)
cat ~/.ssh/id_ed25519.pub

# Change remote to SSH
git remote set-url origin git@github.com:Taikakim/mir-feature-extraction.git
```

### Large Files Warning

If you accidentally committed large files:
```bash
# Remove from git history (be careful!)
git rm --cached <large-file>
git commit --amend
git push --force
```

Better: Ensure `.gitignore` is correct before committing.

---

## Summary

✅ **What you're pushing:**
- Complete MIR framework source code
- Comprehensive documentation
- Setup scripts for external dependencies
- Requirements and configuration

❌ **What you're NOT pushing:**
- Modified external repositories (handled by setup scripts)
- Virtual environments
- Audio files and test data
- Temporary files

🔧 **New users can:**
1. Clone your repo
2. Run setup script
3. Install dependencies
4. Start extracting features immediately

This approach keeps your repository clean, respects external licenses, and makes patches transparent and reproducible.

---

**Ready to push!**
```bash
git remote add origin https://github.com/Taikakim/mir-feature-extraction.git
git push -u origin main
```
