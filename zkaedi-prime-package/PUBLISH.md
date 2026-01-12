# 📦 Publishing to GitHub Packages

## Prerequisites

1. **GitHub Personal Access Token**
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Create a token with `write:packages` and `read:packages` permissions
   - Save it as `GITHUB_TOKEN` environment variable

2. **GitHub Repository**
   - Create a repository (e.g., `zkaedi/zkaedi-prime`)
   - Update `package.json` repository URL

## Publishing Steps

### 1. Configure npm

```bash
# Create/update .npmrc
echo "@zkaedi:registry=https://npm.pkg.github.com" >> .npmrc
echo "//npm.pkg.github.com/:_authToken=${GITHUB_TOKEN}" >> .npmrc
```

### 2. Build Package

```bash
npm run build
```

### 3. Test Locally

```bash
npm test
npm run type-check
```

### 4. Publish

```bash
# Set GitHub token
export GITHUB_TOKEN=your_token_here

# Publish
npm publish
```

## Installing the Package

### In Another Project

1. **Create `.npmrc` in your project:**
   ```
   @zkaedi:registry=https://npm.pkg.github.com
   //npm.pkg.github.com/:_authToken=${GITHUB_TOKEN}
   ```

2. **Install:**
   ```bash
   npm install @zkaedi/zkaedi-prime
   ```

3. **Use:**
   ```typescript
   import { BayesianOptimizer } from "@zkaedi/zkaedi-prime/optimization";
   ```

## Version Management

- Use semantic versioning: `major.minor.patch`
- Update version in `package.json`
- Update `CHANGELOG.md`
- Tag release: `git tag v1.0.0 && git push --tags`

## Troubleshooting

### Authentication Issues
- Ensure `GITHUB_TOKEN` has correct permissions
- Check `.npmrc` configuration
- Verify repository access

### Publishing Errors
- Check package name matches GitHub organization/username
- Verify `package.json` publishConfig
- Ensure version is incremented
