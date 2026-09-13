# physis zsh integration — deterministic workflows + tab autocomplete
# install: echo 'source ~/dev/physis-pro/physis-core/contrib/physis.zsh' >> ~/.zshrc
# or: eval "$(physis-core shell init zsh)"
setopt EXTENDED_HISTORY INC_APPEND_HISTORY SHARE_HISTORY
# completions: physis-core completions zsh > ~/.zsh/completions/_physis-core
# then fpath=(~/.zsh/completions $fpath) before compinit
# auto-learn: every command lands in observation log on next precmd
physis_learn_hook() { (physis-core learn --max 20 >/dev/null 2>&1 &); }
autoload -Uz add-zsh-hook 2>/dev/null && add-zsh-hook precmd physis_learn_hook
