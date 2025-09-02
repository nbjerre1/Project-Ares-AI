using Godot;
using System;

public partial class Xpdrop : Area2D
{
    [Export] public int XpAmount = 1;
    private bool _pickedUp = false; // Guard variable

    // Static counter and limit for simultaneous pickup sounds
    private static int _activePickupSounds = 0;
    private const int MaxSimultaneousPickupSounds = 5;

    public override void _Ready()
    {
        BodyEntered += OnBodyEntered;
    }
    private void DisableShape(CollisionShape2D shape)
    {
        shape.Disabled = true;
    }
    private void OnBodyEntered(Node2D body)
    {
        if (_pickedUp) return; // Prevent double pickup

        if (body is Player player)
        {
            _pickedUp = true; // Set guard
            player.AddXp(XpAmount);

            // Hide the coin and disable collision
            Visible = false;
            SetProcess(false);
            foreach (var child in GetChildren())
            {
                if (child is CollisionShape2D shape)
                    CallDeferred(nameof(DisableShape), shape);
                if (child is CanvasItem item)
                    item.Visible = false;
            }

            // Play the pickup sound if under the limit
            var pickupSound = GetNode<AudioStreamPlayer2D>("PickupSound");
            bool playedSound = false;
            if (pickupSound != null && _activePickupSounds < MaxSimultaneousPickupSounds)
            {
                _activePickupSounds++;
                playedSound = true;
                pickupSound.Play();
            }

            // Delay freeing the node until the sound finishes (if played), otherwise free immediately
            float soundLength = (float)(pickupSound?.Stream?.GetLength() ?? 0.1f);
            if (playedSound)
            {
                GetTree().CreateTimer(soundLength).Timeout += () =>
                {
                    _activePickupSounds--;
                    QueueFree();
                };
            }
            else
            {
                QueueFree();
            }
        }
    }
}
