# YubiKey SSH keys + signed commits (GitHub)

Hardware-backed OpenSSH **security-key** (`ed25519-sk`) on a YubiKey (FIDO2).  
Same key can **authenticate** (`git push`) and **sign** commits (`git commit -S`).

No GPG agent required for this path.

## Prerequisites

- YubiKey with **FIDO2** enabled (default on modern keys)
- `openssh-client` ≥ 8.2 (this box: OpenSSH 9.6)
- `libfido2` installed
- YubiKey plugged in; touch when OpenSSH asks

Optional: `sudo apt install yubikey-manager` for `ykman fido info`.

## 1. Generate a resident (or non-resident) key

```bash
# Non-resident (private handle on disk, still requires YubiKey touch to use):
ssh-keygen -t ed25519-sk \
  -O application=ssh:github-ruffian \
  -C "jasonvanpham@niodoo.com (YubiKey)" \
  -f ~/.ssh/id_ed25519_sk_github

# Optional: resident key stored ON the YubiKey (discoverable):
# ssh-keygen -t ed25519-sk -O resident -O application=ssh:github-ruffian \
#   -C "jasonvanpham@niodoo.com (YubiKey)" -f ~/.ssh/id_ed25519_sk_github
```

Touch the YubiKey when prompted. Set a key passphrase if you want a second factor on disk.

## 2. SSH config (auth for GitHub)

`~/.ssh/config`:

```
Host github.com
  HostName github.com
  User git
  IdentitiesOnly yes
  IdentityFile ~/.ssh/id_ed25519_sk_github
```

## 3. Add the **public** key on GitHub (twice)

```bash
cat ~/.ssh/id_ed25519_sk_github.pub
```

1. **Settings → SSH and GPG keys → New SSH key**  
   - Title: `yubikey-auth`  
   - Key type: **Authentication Key**  
   - Paste `.pub`
2. **New SSH key** again  
   - Title: `yubikey-signing`  
   - Key type: **Signing Key**  
   - Same `.pub` (GitHub allows the same material for both roles)

## 4. Git: sign commits with SSH

```bash
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519_sk_github.pub
git config --global commit.gpgsign true
# optional: tags
git config --global tag.gpgsign true

# Local verify helper (allowed signers)
mkdir -p ~/.config/git
echo "$(git config user.email) $(cat ~/.ssh/id_ed25519_sk_github.pub)" >> ~/.config/git/allowed_signers
git config --global gpg.ssh.allowedSignersFile ~/.config/git/allowed_signers
```

Ensure name/email match what you want on commits:

```bash
git config --global user.name "Jason Van Pham"
git config --global user.email "jasonvanpham@niodoo.com"
```

## 5. Prefer SSH remote

```bash
cd /path/to/hydrodynamic-swarm
git remote set-url origin git@github.com:Ruffian-L/hydrodynamic-swarm.git
ssh -T git@github.com   # touch YubiKey; expect "Hi Ruffian-L!"
```

## 6. Test signed commit

```bash
git commit --allow-empty -S -m "test: yubikey ssh signature"
git log -1 --show-signature
git push origin HEAD
```

On GitHub the commit should show **Verified** (SSH).

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `confirm user presence` hangs | Touch the metal contact |
| `invalid format` / no sk support | Upgrade OpenSSH; ensure `-sk` type |
| Push works, commits unverified | Add the **Signing** key type on GitHub, not only Authentication |
| Wrong key used | `IdentitiesOnly yes` + explicit `IdentityFile` |
| Agent confusion | `ssh-add -l`; `ssh-add -K` for resident keys if needed |

## Security notes

- Private key material for `-sk` is not a normal file secret; operations require the physical key.
- Do not commit `~/.ssh/*` private handles.
- Revoke on GitHub if the YubiKey is lost; regenerate.

---

*Ops note for Hydrodynamic Swarm release hygiene — not part of the science claim surface.*
