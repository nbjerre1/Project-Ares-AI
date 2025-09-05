using Godot;
using System;

public partial class HP : Node
{
    [Export] public int MaxHealth { get; set; } = 100;
    public int CurrentHealth { get; private set; }

    [Signal] public delegate void DiedEventHandler();
    [Signal] public delegate void HealthChangedEventHandler(int newHealth);

    public override void _Ready()
    {
        CurrentHealth = MaxHealth;
    }

    public void TakeDamage(int amount)
    {
        CurrentHealth = Math.Max(CurrentHealth - amount, 0);
        EmitSignal(SignalName.HealthChanged, CurrentHealth);

        if (CurrentHealth <= 0)
            EmitSignal(SignalName.Died);
    }

    public void Heal(int amount)
    {
        CurrentHealth = Math.Min(CurrentHealth + amount, MaxHealth);
        EmitSignal(SignalName.HealthChanged, CurrentHealth);
    }
}
