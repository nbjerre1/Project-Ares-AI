using Godot;
using System;
using System.Collections.Generic;

public partial class Arrow : Area2D
{
    [Export] public float speed = 400f; // Speed of the bullet in units per second
    [Export] float Removerange = 400f; // Maximum distance the bullet can travel
    [Export] public int Damage { get; set; } = 10;
    public PlayerData playerData { get; set; }


    private double travelleddistance = 0;
    private bool hasHit = false;
    public override void _Ready()
    {
        BodyEntered += _on_body_entered;

        var sprite = GetNode<Sprite2D>("Arrow01(32x32)");

        switch (playerData.ActiveArrowUpgrade)
        {
            case PlayerData.ArrowUpgrade.Ice:
                
                var iceTexture = GD.Load<Texture2D>("res://Assets/weapons/icearrow.png");
                Damage += 5;
                sprite.Texture = iceTexture;
                break;
            case PlayerData.ArrowUpgrade.Fire:
                
                var fireTexture = GD.Load<Texture2D>("res://Assets/weapons/firearrow-export.png");
                Damage += 5;
                sprite.Texture = fireTexture;
                break;
        }
    }



    public override void _PhysicsProcess(double delta)
    {

        

        var direction = Vector2.Right.Rotated(Rotation);
        Position += direction * speed * (float)delta; // Move the bullet at a speed of 200 units per second

        travelleddistance += speed * delta; // Update the travelled distance

        if (travelleddistance >= Removerange)
        {
            QueueFree(); // Remove the bullet if it has travelled the maximum distance
        }
    }


    
    private void _on_body_entered(Node body)
    {
        if (hasHit)
            return;
        hasHit = true;

        var healthNode = body.GetNodeOrNull<HP>("Health");
        if (healthNode != null)
        {
            healthNode.TakeDamage(Damage);
        }
        QueueFree();
    }
}
